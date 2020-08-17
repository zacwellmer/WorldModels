# ConvVAE model

import numpy as np
import tensorflow as tf
import os

class CVAE(tf.keras.Model):
    '''
    convolutional variational auto encoder
    '''
    def __init__(self, args):
         
        self.z_size = args.z_size
        self.learning_rate = args.vae_learning_rate
        self.kl_tolerance = args.vae_kl_tolerance
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        super(CVAE, self).__init__()

        self.inference_net_base = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(64, 64, 3)),
            tf.keras.layers.Conv2D(
              filters=32, kernel_size=4, strides=(2, 2), activation='relu',
              name="enc_conv1"),
            tf.keras.layers.Conv2D(
              filters=64, kernel_size=4, strides=(2, 2), activation='relu',
              name="enc_conv2"),
            tf.keras.layers.Conv2D(
              filters=128, kernel_size=4, strides=(2, 2), activation='relu',
              name="enc_conv3"),
            tf.keras.layers.Conv2D(
              filters=256, kernel_size=4, strides=(2, 2), activation='relu',
              name="enc_conv4"),
            tf.keras.layers.Flatten()])
        
        self.mu_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1024)),
            tf.keras.layers.Dense(self.z_size, name="enc_fc_mu")])
        self.logvar_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(1024)),
            tf.keras.layers.Dense(self.z_size, name="enc_fc_log_var")])

        self.generative_net = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(self.z_size,)),
            tf.keras.layers.Dense(units=4*256, activation=tf.nn.relu, name="dec_dense1"),
            tf.keras.layers.Reshape(target_shape=(1, 1, 4*256)),
            tf.keras.layers.Conv2DTranspose(
              filters=128,
              kernel_size=5,
              strides=(2, 2),
              padding="valid",
              activation='relu',
              name="dec_deconv1"),
            tf.keras.layers.Conv2DTranspose(
              filters=64,
              kernel_size=5,
              strides=(2, 2),
              padding="valid",
              activation='relu',
              name="dec_deconv2"),
            tf.keras.layers.Conv2DTranspose(
              filters=32,
              kernel_size=6,
              strides=(2, 2),
              padding="valid",
              activation='relu',
              name="dec_deconv3"),
            tf.keras.layers.Conv2DTranspose(
              filters=3,
              kernel_size=6,
              strides=(2, 2),
              padding="valid",
              activation="sigmoid",
              name="dec_deconv4")])

    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.z_size))
        return self.decode(eps)

    @tf.function # for some reason prevents us from using during training
    def encode(self, x):
        mean, logvar = self.encode_mu_logvar(x)
        z = self.sample_encoding(mean, logvar)
        return z

    def sample_encoding(self, mean, logvar):    
        eps = tf.random.normal(shape=tf.shape(mean))
        z = eps * tf.exp(logvar * .5) + mean
        return z

    def encode_mu_logvar(self, x):
        x = self.inference_net_base(x)
        mean = self.mu_net(x)
        logvar = self.logvar_net(x)
        return mean, logvar

    def decode(self, z):
        probs = self.generative_net(z)
        return probs

    def get_loss(self):
        z_size = self.z_size
        kl_tolerance = self.kl_tolerance
        
        def reconstruction_loss_func(y_true, y_pred):
            # reconstruction loss
            reconstruction_loss = tf.reduce_sum(
              input_tensor=tf.square(y_true - y_pred),
              axis = [1,2,3]
            )
            reconstruction_loss = tf.reduce_mean(input_tensor=reconstruction_loss)
            return reconstruction_loss

        def kl_loss_func(_, y_pred): # _ is where y_true goes, but we don't need it for kl loss
            mean, logvar = y_pred[:, :z_size], y_pred[:, z_size:]

            eps = 1e-6 # avoid taking log of zero
            # augmented kl loss per dim
            kl_loss = - 0.5 * tf.reduce_sum(
              input_tensor=(1 + logvar - tf.square(mean) - tf.exp(logvar)),
              axis = 1
            )
            kl_loss = tf.maximum(kl_loss, kl_tolerance * z_size)
            kl_loss = tf.reduce_mean(input_tensor=kl_loss)

            return kl_loss
        return {'reconstruction': reconstruction_loss_func, 'KL': kl_loss_func}

    def call(self, inputs, training=True):
        return self.__call__(inputs, training)

    def __call__(self, inputs, training=True):
        # not sure why keras forces us to use the trainng flag
        mean, logvar = self.encode_mu_logvar(inputs)
        z = self.sample_encoding(mean, logvar)
        y = self.decode(z)
        mean_and_logvar = tf.concat([mean, logvar], axis=-1)
        return {'reconstruction': y, 'KL': mean_and_logvar}

    def set_random_params(self, stdev=0.5):
        params = self.get_weights()
        rand_params = []
        for param_i in params:
            # David Ha's spicy initialization scheme is wild but from preliminary experiments is critical
            sampled_param = np.random.standard_cauchy(param_i.shape)*stdev / 10000.0 
            rand_params.append(sampled_param) # spice things up
          
        self.set_weights(rand_params)

