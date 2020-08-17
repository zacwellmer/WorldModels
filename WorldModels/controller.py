import numpy as np
import random

import json
import sys

from env import make_env
import time

from rnn.rnn import MDNRNN

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

def make_controller(args):
  # can be extended in the future.
  controller = Controller(args)
  return controller

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

class Controller:
  ''' simple one layer model for car racing '''
  def __init__(self, args):
    self.env_name = args.env_name
    self.exp_mode = args.exp_mode
    self.input_size = args.z_size + args.state_space * args.rnn_size
    self.z_size = args.z_size
    self.a_width = args.a_width
    self.args = args

    if self.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      self.hidden_size = 40
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, self.a_width)
      self.bias_output = np.random.randn(self.a_width)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*self.a_width+self.a_width)
    else:
      self.weight = np.random.randn(self.input_size, self.a_width)
      self.bias = np.random.randn(self.a_width)
      self.param_count = (self.input_size)*self.a_width+self.a_width

    self.render_mode = args.render_mode

  def get_action(self, h):
    '''
    action = np.dot(h, self.weight) + self.bias
    action[0] = np.tanh(action[0])
    action[1] = sigmoid(action[1])
    action[2] = clip(np.tanh(action[2]))
    '''
    if self.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      action = np.tanh(np.dot(h, self.weight) + self.bias)
    
    if 'CarRacing' in self.env_name:
      action[1] = (action[1]+1.0) / 2.0
      action[2] = clip(action[2])

    return action

  def set_model_params(self, model_params):
    if self.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:self.a_width]
      self.weight_output = params_2[self.a_width:].reshape(self.hidden_size, self.a_width)
    else:
      self.bias = np.array(model_params[:self.a_width])
      self.weight = np.array(model_params[self.a_width:]).reshape(self.input_size, self.a_width)

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

  max_episode_length = controller.args.max_frames

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

  for episode in range(num_episode):
    if train_mode: print('episode: {}/{}'.format(episode, num_episode))
    obs = env.reset()

    total_reward = 0.0
    for t in range(max_episode_length):
      if render_mode:
        env.render("human")
      else:
        env.render('rgb_array')

      action = controller.get_action(obs)
      obs, reward, done, info = env.step(action)

      total_reward += reward
      if done:
        break

    if render_mode:
      print("total reward", total_reward, "timesteps", t)
      env.close()
    reward_list.append(total_reward)
    t_list.append(t)
  return reward_list, t_list
