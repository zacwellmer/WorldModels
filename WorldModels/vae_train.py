import sys
import os

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from vae.vae import CVAE
from utils import PARSER

args = PARSER.parse_args()

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

def ds_gen():
    dirname = 'results/{}/{}/record'.format(args.exp_name, args.env_name)
    filenames = os.listdir(dirname)[:10000] # only use first 10k episodes
    n = len(filenames)
    for j, fname in enumerate(filenames):
        if not fname.endswith('npz'): 
            continue
        file_path = os.path.join(dirname, fname)
        with np.load(file_path) as data:
            N = data['obs'].shape[0]
            for i, img in enumerate(data['obs']):
                img_i = img / 255.0
                zerod_outputs = np.zeros([2*args.z_size])
                yield img_i, img_i, zerod_outputs

def create_tf_dataset():
    dataset = tf.data.Dataset.from_generator(ds_gen, output_types=(tf.float32, tf.float32, tf.float32), output_shapes=((64, 64, 3), (64, 64, 3), (args.z_size*2)))
    return dataset

if __name__ == "__main__": 
    model_save_path = "results/{}/{}/tf_vae".format(args.exp_name, args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir)


    dataset_size = 10000 * 1000 # 10k episodes each 1k steps long
    shuffle_size = 20 * 1000 # only loads 20 episodes for shuffle windows b/c im poor and don't have much RAM
    dataset = create_tf_dataset()
    dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True).batch(args.vae_batch_size)

    vae = CVAE(args=args)
    tensorboard_callback.set_model(vae)

    loss_weights = [1.0, 1.0] # weight both the reconstruction and KL loss the same
    vae.compile(optimizer=vae.optimizer, loss=vae.get_loss(), loss_weights=loss_weights)
    step = 0
    n_mb = dataset_size / args.vae_batch_size 
    for i in range(args.vae_num_epoch):
        print('epoch: {}'.format(i))
        j = 0
        for x_batch, targ_batch, blank_batch in dataset:
            j += 1
            step += 1 
           
            loss, recon_loss, reg_loss = vae.train_on_batch(x=x_batch, y=[targ_batch, blank_batch])
            with summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=step)
                tf.summary.scalar('reconstruction loss', loss, step=step)
                tf.summary.scalar('regularization loss', loss, step=step)
            if j % 100 == 0:
                print('epoch: {} mb: {}/{} loss: {} reconstruction loss: {} regularization loss: {}'.format(i, j, n_mb, loss, recon_loss, reg_loss))
                sys.stdout.flush()
        print('saving')
        tf.keras.models.save_model(vae, model_save_path, include_optimizer=True, save_format='tf')
