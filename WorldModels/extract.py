'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym

from env import make_env
from model import make_model

MAX_FRAMES = 1000 # max length of carracing
MAX_TRIALS = 140 # just use this to extract one trial. 

render_mode = False # for debugging.

DIR_NAME = 'results/eager/record'
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)

controller = make_model()

total_frames = 0
env = make_env(render_mode=render_mode, full_episode=False, with_obs=True)
for trial in range(MAX_TRIALS): # 200 trials per worker
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = DIR_NAME+"/"+str(random_generated_int)+".npz"
    recording_frame = []
    recording_action = []
    recording_reward = []
    recording_done = []

    np.random.seed(random_generated_int)
    env.seed(random_generated_int)

    # random policy
    controller.init_random_model_params(stdev=np.random.rand()*0.01)

    tot_r = 0
    [obs, frame] = env.reset() # pixels

    for i in range(MAX_FRAMES):
      if render_mode:
        env.render("human")
      else:
        env.render("rgb_array")

      recording_frame.append(frame)
      action = controller.get_action(obs)

      recording_action.append(action)

      [obs, frame], reward, done, info = env.step(action)
      tot_r += reward

      recording_reward.append(reward)
      recording_done.append(done)

      if done:
        print('total reward {}'.format(tot_r))
        break

    total_frames += (i+1)
    print('total reward {}'.format(tot_r))
    print("dead at", i+1, "total recorded frames for this worker", total_frames)
    recording_frame = np.array(recording_frame, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)
    recording_done = np.array(recording_done, dtype=np.bool)

    np.savez_compressed(filename, obs=recording_frame, action=recording_action, reward=recording_reward, done=recording_done)
  except gym.error.Error:
    print("stupid gym error, life goes on")
    env.close()
    env = make_env(render_mode=render_mode, full_episode=False, with_obs=True)
    continue
env.close()
