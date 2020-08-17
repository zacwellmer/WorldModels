import numpy as np
import gym
import json
import os
import tensorflow as tf
import gc

from PIL import Image
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

class CarRacingWrapper(CarRacing):
  def __init__(self, full_episode=False):
    super(CarRacingWrapper, self).__init__()
    self.full_episode = full_episode
    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3)) # , dtype=np.uint8

  def _process_frame(self, frame):
    obs = frame[0:84, :, :]
    obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
    obs = np.array(obs)
    return obs

  def _step(self, action):
    obs, reward, done, _ = super(CarRacingWrapper, self)._step(action)
    if self.full_episode:
      return self._process_frame(obs), reward, False, {}
    return self._process_frame(obs), reward, done, {}

from vae.vae import CVAE
from rnn.rnn import MDNRNN, rnn_next_state, rnn_init_state
class CarRacingMDNRNN(CarRacingWrapper):
  def __init__(self, args, load_model=True, full_episode=False, with_obs=False):
    super(CarRacingMDNRNN, self).__init__(full_episode=full_episode)
    self.with_obs = with_obs # whether or not to return the frame with the encodings
    self.vae = CVAE(args)
    self.rnn = MDNRNN(args)
     
    if load_model:
      self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
      self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])
    self.rnn_states = rnn_init_state(self.rnn)
    
    self.full_episode = False 
    self.observation_space = Box(low=np.NINF, high=np.Inf, shape=(args.z_size+args.rnn_size*args.state_space))
  def encode_obs(self, obs):
    # convert raw obs to z, mu, logvar
    result = np.copy(obs).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    z = self.vae.encode(result)[0]
    return z
  def reset(self):
    self.rnn_states = rnn_init_state(self.rnn)
    if self.with_obs:
        [z_state, obs] = super(CarRacingMDNRNN, self).reset() # calls step
        self.N_tiles = len(self.track)
        return [z_state, obs]
    else:
        z_state = super(CarRacingMDNRNN, self).reset() # calls step
        self.N_tiles = len(self.track)
        return z_state
  def _step(self, action):
    obs, reward, done, _ = super(CarRacingMDNRNN, self)._step(action)
    z = tf.squeeze(self.encode_obs(obs))
    h = tf.squeeze(self.rnn_states[0])
    c = tf.squeeze(self.rnn_states[1])
    if self.rnn.args.state_space == 2:
        z_state = tf.concat([z, c, h], axis=-1)
    else:
        z_state = tf.concat([z, h], axis=-1)
    if action is not None: # don't compute state on reset
        self.rnn_states = rnn_next_state(self.rnn, z, action, self.rnn_states)
    if self.with_obs:
        return [z_state, obs], reward, done, {}
    else:
        return z_state, reward, done, {}
  def close(self):
    super(CarRacingMDNRNN, self).close()
    tf.keras.backend.clear_session()
    gc.collect()

from ppaquette_gym_doom.doom_take_cover import DoomTakeCoverEnv
from gym.utils import seeding
class DoomTakeCoverMDNRNN(DoomTakeCoverEnv):
  def __init__(self, args, render_mode=False, load_model=True, with_obs=False):
    super(DoomTakeCoverMDNRNN, self).__init__()

    self.with_obs = with_obs

    self.no_render = True
    if render_mode:
      self.no_render = False
    self.current_obs = None

    self.vae = CVAE(args)
    self.rnn = MDNRNN(args)

    if load_model:
      self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
      self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])

    self.action_space = Box(low=-1.0, high=1.0, shape=())
    self.obs_size = self.rnn.args.z_size + self.rnn.args.rnn_size * self.rnn.args.state_space

    self.observation_space = Box(low=0, high=255, shape=(64, 64, 3))
    self.actual_observation_space = Box(low=-50., high=50., shape=(self.obs_size))

    self._seed()

    self.rnn_states = None
    self.z = None
    self.restart = None
    self.frame_count = None
    self.viewer = None
    self._reset()

  def close(self):
    super(DoomTakeCoverMDNRNN, self).close()
    tf.keras.backend.clear_session()
    gc.collect()

  def _step(self, action):

    # update states of rnn
    self.frame_count += 1
   
    self.rnn_states = rnn_next_state(self.rnn, self.z, action, self.rnn_states) 

    # actual action in wrapped env:

    threshold = 0.3333
    full_action = [0] * 43

    if action < -threshold:
      full_action[11] =1

    if action > threshold:
      full_action[10] = 1

    obs, reward, done, _ = super(DoomTakeCoverMDNRNN, self)._step(full_action)
    small_obs = self._process_frame(obs)
    self.current_obs = small_obs
    self.z = self._encode(small_obs)

    if done:
      self.restart = 1
    else:
      self.restart = 0

    if self.with_obs:
      return [self._current_state(), self.current_obs], reward, done, {}
    else:
      return self._current_state(), reward, done, {}

  def _encode(self, img):
    simple_obs = np.copy(img).astype(np.float)/255.0
    simple_obs = simple_obs.reshape(1, 64, 64, 3)
    z = self.vae.encode(simple_obs)[0]
    return z

  def _reset(self):
    obs = super(DoomTakeCoverMDNRNN, self)._reset()
    small_obs = self._process_frame(obs)
    self.current_obs = small_obs
    self.rnn_states = rnn_init_state(self.rnn)
    self.z = self._encode(small_obs)
    self.restart = 1
    self.frame_count = 0

    if self.with_obs:
      return [self._current_state(), self.current_obs]
    else:
      return self._current_state()

  def _process_frame(self, frame):
    obs = frame[0:400, :, :]
    obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
    obs = np.array(obs)
    return obs

  def _current_state(self):
    if self.rnn.args.state_space == 2:
      return np.concatenate([self.z, tf.keras.backend.flatten(self.rnn_states[1]), tf.keras.backend.flatten(self.rnn_states[0])], axis=0) # cell then hidden fro some reason
    return np.concatenate([self.z, tf.keras.backend.flatten(self.rnn_states[0])], axis=0) # only the hidden state

  def _seed(self, seed=None):
    if seed:
      tf.random.set_seed(seed)
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

from rnn.rnn import rnn_sim
class DreamDoomTakeCoverMDNRNN:
  def __init__(self, args, render_mode=False, load_model=True):

    self.render_mode = render_mode
    model_path_name = 'results/{}/{}'.format(args.exp_name, args.env_name)
    with open(os.path.join(model_path_name, 'tf_initial_z/initial_z.json'), 'r') as f:
      [initial_mu, initial_logvar] = json.load(f)

    self.initial_mu_logvar = np.array([list(elem) for elem in zip(initial_mu, initial_logvar)])

    self.vae = CVAE(args)
    self.rnn = MDNRNN(args)

    if load_model:
      self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
      self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load('results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])

    # future versions of OpenAI gym needs a dtype=np.float32 in the next line:
    self.action_space = Box(low=-1.0, high=1.0, shape=())
    obs_size = self.rnn.args.z_size + self.rnn.args.rnn_size * self.rnn.args.state_space
    # future versions of OpenAI gym needs a dtype=np.float32 in the next line:
    self.observation_space = Box(low=-50., high=50., shape=(obs_size,))

    self.rnn_states = None
    self.o = None

    self._training=True    

    self.seed() 
    self.reset()

  def _sample_init_z(self):
    idx = self.np_random.randint(low=0, high=self.initial_mu_logvar.shape[0])
    init_mu, init_logvar = self.initial_mu_logvar[idx]
    init_mu = init_mu / 10000.0
    init_logvar = init_logvar / 10000.0
    init_z = init_mu + np.exp(init_logvar/2.0) * self.np_random.randn(*init_logvar.shape)
    return init_z

  def reset(self):
    self.rnn_states = rnn_init_state(self.rnn)
    z = np.expand_dims(self._sample_init_z(), axis=0)
    self.o = z
    z_ch = tf.concat([z, self.rnn_states[1], self.rnn_states[0]], axis=-1)
    return tf.squeeze(z_ch)

  def seed(self, seed=None):
    if seed:
      tf.random.set_seed(seed)
   
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    rnn_states_p1, z_tp1, r_tp1, d_tp1 = rnn_sim(self.rnn, self.o, self.rnn_states, action, training=self._training)
    self.rnn_states = rnn_states_p1
    self.o = z_tp1

    z_ch = tf.squeeze(tf.concat([z_tp1, self.rnn_states[1], self.rnn_states[0]], axis=-1))    
    return z_ch.numpy(), tf.squeeze(r_tp1), d_tp1.numpy(), {}

  def close(self):
    tf.keras.backend.clear_session()
    gc.collect()

  def render(self, mode):
    pass

def make_env(args, dream_env=False, seed=-1, render_mode=False, full_episode=False, with_obs=False, load_model=True):
  if args.env_name == 'DoomTakeCover-v0':
    if dream_env:
      print('making rnn doom environment')
      env = DreamDoomTakeCoverMDNRNN(args=args, render_mode=render_mode, load_model=load_model)
    else:
      print('making real doom environment')
      env = DoomTakeCoverMDNRNN(args=args, render_mode=render_mode, load_model=load_model, with_obs=with_obs)
  else:
    if dream_env:
      raise ValueError('training in dreams for carracing is not yet supported')
    else:
      print('makeing real CarRacing environment')
      env = CarRacingMDNRNN(args=args, full_episode=full_episode, with_obs=with_obs, load_model=load_model)
  if (seed >= 0):
    env.seed(seed)
  return env
