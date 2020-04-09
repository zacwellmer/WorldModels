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

from vae.vae import CVAE
from rnn.rnn import HyperParams, MDNRNN, sample_vae

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

DATA_DIR = "results/eager/series"
model_save_path = "results/eager/tf_rnn"
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
  
def random_batch():
  indices = np.random.permutation(N_data)[0:batch_size]
  mu = data_mu[indices]
  logvar = data_logvar[indices]
  action = data_action[indices]
  z = sample_vae(mu, logvar)
  return z, action 

def default_hps():
  return HyperParams(num_steps=4000,
                     max_seq_len=999, # train on sequences of 1000 (so 999 + teacher forcing shift)
                     input_seq_width=35,    # width of our data (32 + 3 actions)
                     output_seq_width=32,    # width of our data is 32
                     rnn_size=256,    # number of rnn cells
                     batch_size=100,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

hps_model = default_hps()
hps_sample = hps_model._replace(batch_size=1, max_seq_len=1, use_recurrent_dropout=0, is_training=0)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
max_seq_len = hps_model.max_seq_len

N_data = len(data_mu) # should be 10k
batch_size = hps_model.batch_size

rnn = MDNRNN(hps_model)
rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss())

# train loop:
hps = hps_model
start = time.time()
for step in range(hps.num_steps):
    curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate
    rnn.optimizer.learning_rate = curr_learning_rate

    raw_z, raw_a = random_batch()
    inputs =tf.concat([raw_z[:, :-1, :], raw_a[:, :-1, :]], axis=2)
    outputs = raw_z[:, 1:, :]

    loss = rnn.train_on_batch(x=inputs, y=outputs)
    #with tf.GradientTape() as tape:
    #    pred = rnn(inputs=inputs, training=True)
    #    loss = rnn.loss_fn(y_true=outputs, y_pred=pred)
    #grads = tape.gradient(loss, rnn.trainable_weights)
    #rnn.optimizer.apply_gradients(zip(grads, rnn.trainable_weights))

    if (step%20==0 and step > 0):
        end = time.time()
        time_taken = end-start
        start = time.time()
        output_log = "step: %d, lr: %.6f, loss: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, loss, time_taken)
        print(output_log)

        tf.keras.models.save_model(rnn, model_save_path, include_optimizer=True, save_format='tf')
