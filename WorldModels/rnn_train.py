'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time

from rnn.rnn import MDNRNN, sample_vae
from utils import PARSER
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
tf.config.experimental_run_functions_eagerly # used for debugging

args = PARSER.parse_args()

DATA_DIR = "results/{}/{}/series".format(args.exp_name, args.env_name)
model_save_path = "results/{}/{}/tf_rnn".format(args.exp_name, args.env_name)
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
data_d = raw_data["done"]
N_data = len(data_mu) # should be 10k

# save 1000 initial mu and logvars. Used for sampling when training in dreams
initial_z_save_path = "results/{}/{}/tf_initial_z".format(args.exp_name, args.env_name)
if not os.path.exists(initial_z_save_path):
  os.makedirs(initial_z_save_path)
initial_mu = []
initial_logvar = []
for i in range(1000):
  mu = np.copy(data_mu[i][0, :]*10000).astype(np.int).tolist()
  logvar = np.copy(data_logvar[i][0, :]*10000).astype(np.int).tolist()
  initial_mu.append(mu)
  initial_logvar.append(logvar)
with open(os.path.join(initial_z_save_path, "initial_z.json"), 'wt') as outfile:
  json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))

def random_batch():
  indices = np.random.permutation(N_data)[0:args.rnn_batch_size]
  # suboptimal b/c we are always only taking first set of steps
  mu = data_mu[indices][:, :args.rnn_max_seq_len] 
  logvar = data_logvar[indices][:, :args.rnn_max_seq_len]
  action = data_action[indices][:, :args.rnn_max_seq_len]
  z = sample_vae(mu, logvar)
  d = tf.cast(data_d[indices], tf.float16)[:, :args.rnn_max_seq_len]
  return z, action, d

rnn = MDNRNN(args=args)
rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss())

# train loop:
start = time.time()
step = 0
for step in range(args.rnn_num_steps):
    curr_learning_rate = (args.rnn_learning_rate-args.rnn_min_learning_rate) * (args.rnn_decay_rate) ** step + args.rnn_min_learning_rate
    rnn.optimizer.learning_rate = curr_learning_rate
    
    raw_z, raw_a, raw_d = random_batch()

    inputs = tf.concat([raw_z, raw_a], axis=2)

    dummy_zero = tf.zeros([raw_z.shape[0], 1, raw_z.shape[2]], dtype=tf.float16)
    z_targ = tf.concat([raw_z[:, 1:, :], dummy_zero], axis=1) # zero pad the end but we don't actually use it
    z_mask = 1.0 - raw_d
    z_targ = tf.concat([z_targ, z_mask], axis=2) # use a signal to not pass grad

    if args.env_name == 'CarRacing-v0':
      outputs = z_targ
    else:
      d_mask = tf.concat([tf.ones([args.rnn_batch_size, 1, 1], dtype=tf.float16), 1.0 - raw_d[:, :-1, :]], axis=1)
      d_targ = tf.concat([raw_d, d_mask], axis=2)
      outputs = [z_targ, d_targ]

    loss = rnn.train_on_batch(x=inputs, y=outputs)

    if (step%20==0 and step > 0):
        end = time.time()
        time_taken = end-start
        start = time.time()
        if args.env_name == 'CarRacing-v0':
          output_log = "step: %d, lr: %.6f, loss: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, loss, time_taken)
        else:
          output_log = "step: %d, lr: %.6f, loss: %.4f, z_loss: %.4f, d_loss: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, loss[0], loss[1], loss[2], time_taken)
        print(output_log)

        tf.keras.models.save_model(rnn, model_save_path, include_optimizer=True, save_format='tf')
