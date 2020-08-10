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
                yield img_i
if __name__ == "__main__": 
    model_save_path = "results/{}/{}/tf_vae".format(args.exp_name, args.env_name)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    tensorboard_dir = os.path.join(model_save_path, 'tensorboard')
    summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    summary_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, write_graph=False)
    shuffle_size = 20 * 1000 # only loads ~20 episodes for shuffle windows b/c im poor and don't have much RAM
    ds = tf.data.Dataset.from_generator(ds_gen, output_types=tf.float32, output_shapes=(64, 64, 3))
    ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=True).batch(args.vae_batch_size)
    ds = ds.prefetch(100) # prefetch 100 batches in the buffer #tf.data.experimental.AUTOTUNE)
    vae = CVAE(args=args)
    tensorboard_callback.set_model(vae)
    loss_weights = [1.0, 1.0] # weight both the reconstruction and KL loss the same
    vae.compile(optimizer=vae.optimizer, loss=vae.get_loss(), loss_weights=loss_weights)
    step = 0
    blank_batch = np.zeros([2*args.z_size])
    for i in range(args.vae_num_epoch):
        j = 0
        for x_batch in ds:
            if i == 0 and j == 0:
                vae._set_inputs(x_batch)
            j += 1
            step += 1 
           
            loss = vae.train_on_batch(x=x_batch, y={'reconstruction': x_batch, 'KL': blank_batch}, return_dict=True)
            [tf.summary.scalar(loss_key, loss_val, step=step) for loss_key, loss_val in loss.items()] 
            if j % 100 == 0:
                output_log = 'epoch: {} mb: {}'.format(i, j)
                for loss_key, loss_val in loss.items():
                    output_log += ', {}: {:.4f}'.format(loss_key, loss_val)
                print(output_log)
        print('saving')
        tf.keras.models.save_model(vae, model_save_path, include_optimizer=True, save_format='tf')
