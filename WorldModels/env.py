import numpy as np
import gym
import tensorflow as tf

from PIL import Image
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

SCREEN_X = 64
SCREEN_Y = 64

def _process_frame(frame):
  obs = frame[0:84, :, :]
  obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
  obs = np.array(obs)
  return obs

class CarRacingWrapper(CarRacing):
  def __init__(self, full_episode=False):
    super(CarRacingWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3)) # , dtype=np.uint8

  def _step(self, action):
    obs, reward, done, _ = super(CarRacingWrapper, self)._step(action)
    if self.full_episode:
      return _process_frame(obs), reward, False, {}
    return _process_frame(obs), reward, done, {}

from vae.vae import CVAE
from rnn.rnn import MDNRNN, hps_sample, rnn_next_state, rnn_init_state
class CarRacingMDNRNN(CarRacingWrapper):
  def __init__(self, load_model=True, full_episode=False, with_obs=False):
    super(CarRacingMDNRNN, self).__init__(full_episode=full_episode)
    self.with_obs = with_obs # whether or not to return the frame with the encodings
    self.vae = CVAE(batch_size=1)
    self.rnn = MDNRNN(hps_sample)
     
    if load_model:
      self.vae.set_weights(tf.keras.models.load_model('results/eager/tf_vae', compile=False).get_weights())
      self.rnn.set_weights(tf.keras.models.load_model('results/eager/tf_rnn', compile=False).get_weights())

    self.rnn_states = rnn_init_state(self.rnn)
    
    self.full_episode = False 
    self.observation_space = Box(low=np.NINF, high=np.Inf, shape=(32+256))

  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    mu, logvar = self.vae.encode_mu_logvar(result)
    mu = mu[0]
    logvar = logvar[0]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
    return z, mu, logvar

  def reset(self):
    self.rnn_states = rnn_init_state(self.rnn)
    if self.with_obs:
        [z_h, obs] = super(CarRacingWrapper, self).reset() # calls step
        return [z_h, obs]
    else:
        z_h = super(CarRacingWrapper, self).reset() # calls step
        return z_h

  def _step(self, action):
    obs, reward, done, _ = super(CarRacingMDNRNN, self)._step(action)
    z, _, _ = self.encode_obs(obs)
    h = tf.squeeze(self.rnn_states[0])
    z_h = tf.concat([z, h], axis=-1)

    if action is not None: # don't compute state on reset
        self.rnn_states = rnn_next_state(self.rnn, z, action, self.rnn_states)
    if self.with_obs:
        return [z_h, obs], reward, done, {}
    else:
        return z_h, reward, done, {}

def make_env(seed=-1, render_mode=False, full_episode=False, with_obs=False):
  env = CarRacingMDNRNN(full_episode=full_episode, with_obs=with_obs)
  if (seed >= 0):
    env.seed(seed)
  return env

# from https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
if __name__=="__main__":
  from pyglet.window import key
  a = np.array( [0.0, 0.0, 0.0] )
  def key_press(k, mod):
    global restart
    if k==0xff0d: restart = True
    if k==key.LEFT:  a[0] = -1.0
    if k==key.RIGHT: a[0] = +1.0
    if k==key.UP:    a[1] = +1.0
    if k==key.DOWN:  a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
  def key_release(k, mod):
    if k==key.LEFT  and a[0]==-1.0: a[0] = 0
    if k==key.RIGHT and a[0]==+1.0: a[0] = 0
    if k==key.UP:    a[1] = 0
    if k==key.DOWN:  a[2] = 0
  env = CarRacing()
  env.render()
  env.viewer.window.on_key_press = key_press
  env.viewer.window.on_key_release = key_release
  while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
      s, r, done, info = env.step(a)
      total_reward += r
      if steps % 200 == 0 or done:
        print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
        print("step {} total_reward {:+0.2f}".format(steps, total_reward))
      steps += 1
      env.render()
      if done or restart: break
  env.monitor.close()
