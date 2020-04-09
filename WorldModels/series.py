'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
from vae.vae import CVAE

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

DATA_DIR = "results/eager/record"
SERIES_DIR = "results/eager/series"
model_path_name = "results/eager/tf_vae"

if not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)
Z_SIZE = 32
def ds_gen():
    dirname = 'results/eager/record'
    filenames = os.listdir(dirname)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(dirname, fname)
        with np.load(file_path) as data:
            N = data['obs'].shape[0]
            for i, img in enumerate(data['obs']):
                action = data['action'][i]
                yield img, action

def create_tf_dataset():
    dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float32, tf.float32), output_shapes=((64, 64, 3), (3)))
    return dataset

@tf.function
def encode_batch(batch_img):
  simple_obs = batch_img/255.0
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  return mu, logvar

def decode_batch(batch_z):
  # decode the latent vector
  batch_img = vae.decode(z.reshape(batch_size, z_size)) * 255.
  batch_img = np.round(batch_img).astype(np.uint8)
  batch_img = batch_img.reshape(batch_size, 64, 64, 3)
  return batch_img


# Hyperparameters for ConvVAE
z_size=32
batch_size=1000 # treat every episode as a batch of 1000!
learning_rate=0.0001
kl_tolerance=0.5

filelist = os.listdir(DATA_DIR)
filelist.sort()
filelist = filelist[0:10000]
dataset = create_tf_dataset()
dataset = dataset.batch(batch_size, drop_remainder=True)

vae = CVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance)

vae.set_weights(tf.keras.models.load_model(model_path_name, compile=False).get_weights())
mu_dataset = []
logvar_dataset = []
action_dataset = []

i=0
for obs_batch, action_batch in dataset:
  i += 1
  mu, logvar = encode_batch(obs_batch)

  mu_dataset.append(mu.numpy().astype(np.float16))
  logvar_dataset.append(logvar.numpy().astype(np.float16))
  action_dataset.append(action_batch.numpy())

  if ((i+1) % 100 == 0):
    print(i+1)

action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
