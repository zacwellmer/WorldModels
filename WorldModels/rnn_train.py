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
#tf.config.experimental_run_functions_eagerly(True) # used for debugging

args = PARSER.parse_args()

DATA_DIR = "results/{}/{}/series".format(args.exp_name, args.env_name)
model_save_path = "results/{}/{}/tf_rnn".format(args.exp_name, args.env_name)
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
with open(model_save_path + '/args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)
raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
data_r = raw_data["reward"]
data_d = raw_data["done"]
data_N = raw_data["N"]
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

def ds_gen():
  for _ in range(args.rnn_num_steps):
    indices = np.random.permutation(N_data)[0:args.rnn_batch_size]
    # suboptimal b/c we are always only taking first set of steps
    mu = data_mu[indices][:, :args.rnn_max_seq_len] 
    logvar = data_logvar[indices][:, :args.rnn_max_seq_len]
    action = data_action[indices][:, :args.rnn_max_seq_len]
    z = sample_vae(mu, logvar)
    r = tf.cast(data_r[indices], tf.float16)[:, :args.rnn_max_seq_len]
    d = tf.cast(data_d[indices], tf.float16)[:, :args.rnn_max_seq_len]
    N = tf.cast(data_N[indices], tf.float16)[:, :args.rnn_max_seq_len]
    yield z, action, r, d, N
    
dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float16, tf.float16, tf.float16, tf.float16, tf.float16), \
    output_shapes=((args.rnn_batch_size, args.rnn_max_seq_len, args.z_size), \
    (args.rnn_batch_size, args.rnn_max_seq_len, args.a_width), \
    (args.rnn_batch_size, args.rnn_max_seq_len, 1), \
    (args.rnn_batch_size, args.rnn_max_seq_len, 1), \
    (args.rnn_batch_size, args.rnn_max_seq_len, 1)))
dataset = dataset.prefetch(10)
tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
summary_writer = tf.summary.create_file_writer(tensorboard_dir)
summary_writer.set_as_default()
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=False)

rnn = MDNRNN(args=args)
rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss())
tensorboard_callback.set_model(rnn)

# train loop:
start = time.time()
step = 0
for raw_z, raw_a, raw_r, raw_d, raw_N in dataset:
    curr_learning_rate = (args.rnn_learning_rate-args.rnn_min_learning_rate) * (args.rnn_decay_rate) ** step + args.rnn_min_learning_rate
    rnn.optimizer.learning_rate = curr_learning_rate
    
    inputs = tf.concat([raw_z, raw_a], axis=2)

    if step == 0:
        rnn._set_inputs(inputs)

    dummy_zero = tf.zeros([raw_z.shape[0], 1, raw_z.shape[2]], dtype=tf.float16)
    z_targ = tf.concat([raw_z[:, 1:, :], dummy_zero], axis=1) # zero pad the end but we don't actually use it
    z_mask = 1.0 - raw_d
    z_targ = tf.concat([z_targ, z_mask], axis=2) # use a signal to not pass grad

    outputs = {'MDN': z_targ}
    if args.rnn_r_pred == 1:
        r_mask = tf.concat([tf.ones([args.rnn_batch_size, 1, 1], dtype=tf.float16), 1.0 - raw_d[:, :-1, :]], axis=1)
        r_targ = tf.concat([raw_r, r_mask], axis=2)
        outputs['r'] = r_targ
    if args.rnn_d_pred == 1:
        d_mask = tf.concat([tf.ones([args.rnn_batch_size, 1, 1], dtype=tf.float16), 1.0 - raw_d[:, :-1, :]], axis=1)
        d_targ = tf.concat([raw_d, d_mask], axis=2)
        outputs['d'] = d_targ
    loss = rnn.train_on_batch(x=inputs, y=outputs, return_dict=True)
    [tf.summary.scalar(loss_key, loss_val, step=step) for loss_key, loss_val in loss.items()]

    if (step%20==0 and step > 0):
        end = time.time()
        time_taken = end-start
        start = time.time()
        output_log = "step: %d, train_time_taken: %.4f, lr: %.6f" % (step, time_taken, curr_learning_rate)
        for loss_key, loss_val in loss.items():
            output_log += ', {}: {:.4f}'.format(loss_key, loss_val)
        print(output_log)
    if (step%1000==0 and step > 0):
        tf.keras.models.save_model(rnn, model_save_path, include_optimizer=True, save_format='tf')
    step += 1
