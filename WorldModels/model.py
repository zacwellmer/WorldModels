import numpy as np
import random

import json
import sys

from env import make_env, CarRacingMDNRNN
import time

from rnn.rnn import hps_sample, MDNRNN, rnn_init_state, rnn_next_state, rnn_output, rnn_output_size

render_mode = True

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

EXP_MODE = MODE_ZH

def make_model():
  # can be extended in the future.
  controller = Controller()
  return controller

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

def passthru(x):
  return x

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def sample(p):
  return np.argmax(np.random.multinomial(1, p))

class Controller:
  ''' simple one layer model for car racing '''
  def __init__(self):
    self.env_name = "carracing"

    self.input_size = rnn_output_size(EXP_MODE)
    self.z_size = 32

    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, 3)
      self.bias_output = np.random.randn(3)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*3+3)
    else:
      self.weight = np.random.randn(self.input_size, 3)
      self.bias = np.random.randn(3)
      self.param_count = (self.input_size)*3+3

    self.render_mode = False

  def get_action(self, h):
    '''
    action = np.dot(h, self.weight) + self.bias
    action[0] = np.tanh(action[0])
    action[1] = sigmoid(action[1])
    action[2] = clip(np.tanh(action[2]))
    '''
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      action = np.tanh(np.dot(h, self.weight) + self.bias)
    
    action[1] = (action[1]+1.0) / 2.0
    action[2] = clip(action[2])

    return action

  def set_model_params(self, model_params):
    if EXP_MODE == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:3]
      self.weight_output = params_2[3:].reshape(self.hidden_size, 3)
    else:
      self.bias = np.array(model_params[:3])
      self.weight = np.array(model_params[3:]).reshape(self.input_size, 3)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)

def simulate(controller, env, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):

  reward_list = []
  t_list = []

  max_episode_length = 1000
  recording_mode = False
  penalize_turning = False

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

  for episode in range(num_episode):

    obs = env.reset()

    total_reward = 0.0

    random_generated_int = np.random.randint(2**31-1)

    filename = "record/"+str(random_generated_int)+".npz"

    for t in range(max_episode_length):

      if render_mode:
        env.render("human")
      else:
        env.render('rgb_array')

      action = controller.get_action(obs)


      obs, reward, done, info = env.step(action)

      extra_reward = 0.0 # penalize for turning too frequently
      if train_mode and penalize_turning:
        extra_reward -= np.abs(action[0])/10.0
        reward += extra_reward

      #if (render_mode):
      #  print("action", action, "step reward", reward)
      total_reward += reward

      if done:
        break

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list

def main():

  assert len(sys.argv) > 1, 'python model.py render/norender path_to_mode.json [seed]'


  render_mode_string = str(sys.argv[1])
  if (render_mode_string == "render"):
    render_mode = True
  else:
    render_mode = False

  use_model = False
  if len(sys.argv) > 2:
    use_model = True
    filename = sys.argv[2]
    print("filename", filename)

  the_seed = np.random.randint(10000)
  if len(sys.argv) > 3:
    the_seed = int(sys.argv[3])
    print("seed", the_seed)

  if (use_model):
    model = make_model()
    print('model size', model.param_count)
    env = make_env(render_mode=render_mode)
    model.load_model(filename)
  else:
    model = make_model(load_model=False)
    print('model size', model.param_count)
    model.make_env(render_mode=render_mode)
    model.init_random_model_params(stdev=np.random.rand()*0.01)

  N_episode = 100
  if render_mode:
    N_episode = 1
  reward_list = []
  for i in range(N_episode):
    reward, steps_taken = simulate(model, env,
      train_mode=False, render_mode=render_mode, num_episode=1)
    if render_mode:
      print("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)
    else:
      print(reward[0])
    reward_list.append(reward[0])
  if not render_mode:
    print("seed", the_seed, "average_reward", np.mean(reward_list), "stdev", np.std(reward_list))

if __name__ == "__main__":
  main()
