import numpy as np
from collections import namedtuple
import json
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

# hyperparameters for our model. I was using an older tf version, when HParams was not available ...

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                        ])

def default_hps():
  return HyperParams(num_steps=2000, # train model for 2000 steps.
                     max_seq_len=1000, # train on sequences of 100
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

def sample_vae(vae_mu, vae_logvar):
    sz = vae_mu.shape[1]
    mu_logvar = tf.concat([vae_mu, vae_logvar], axis=1)
    z = tfp.layers.DistributionLambda(lambda theta: tfp.distributions.MultivariateNormalDiag(loc=theta[:, :sz], scale_diag=tf.exp(theta[:, sz:])), dtype=tf.float16)
    return z(mu_logvar)

class MDNRNN(tf.keras.Model):
    def __init__(self, hps):
        super(MDNRNN, self).__init__()
        self.hps = hps
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.hps.learning_rate, clipvalue=self.hps.grad_clip)

        self.loss_fn = self.get_loss() 

        lstm_cell = tf.keras.layers.LSTMCell(units=hps.rnn_size)

        self.inference_base = tf.keras.layers.RNN(cell=lstm_cell, return_sequences=True, return_state=True, time_major=False)

        self.out_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.hps.rnn_size),
            tf.keras.layers.Dense(3 * hps.num_mixture * hps.output_seq_width, name="mu_logstd_logmix_net")])

        super(MDNRNN, self).build((self.hps.batch_size, self.hps.max_seq_len, self.hps.input_seq_width))

    @tf.function
    def get_output_size(self, mode):
        if mode == MODE_ZCH:
            return (32+256+256)
        if (mode == MODE_ZC) or (mode == MODE_ZH):
            return (32+256)
        return 32 # MODE_Z or MODE_Z_HIDDEN

    def get_loss(self):
        num_mixture = self.hps.num_mixture
        output_seq_width = self.hps.output_seq_width
        """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
        # Construct a loss function with the right number of mixtures and outputs
        def DHA_loss_func(y_true, y_pred):
            '''
            This loss function is defined for N*k components each containing a gaussian of 1 feature
            '''
            # Reshape inputs in case this is used in a TimeDistribued layer
            y_pred = tf.reshape(y_pred, [-1, 3*num_mixture], name='reshape_ypreds')
            vae_z = tf.reshape(y_true, [-1, 1], name='reshape_ytrue')
            
            out_mu, out_logstd, out_logpi = tf.split(y_pred, num_or_size_splits=3, axis=1, name='mdn_coef_split')
            out_logpi = out_logpi - tf.reduce_logsumexp(input_tensor=out_logpi, axis=1, keepdims=True) # normalize

            logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
            lognormal = -0.5 * ((vae_z - out_mu) / tf.exp(out_logstd)) ** 2 - out_logstd - logSqrtTwoPI
            v = out_logpi + lognormal
            v = tf.reduce_logsumexp(input_tensor=v, axis=1, keepdims=True)
            return -tf.reduce_mean(input_tensor=v)
        return DHA_loss_func

    def set_random_params(self, stdev=0.5):
        params = self.get_weights()
        rand_params = []
        for param_i in params:
            # David's spicy initialization scheme is wild but from preliminary experiments is critical
            sampled_param = np.random.standard_cauchy(param_i.shape)*stdev / 10000.0 
            rand_params.append(sampled_param) # spice things up
          
        self.set_weights(rand_params)
    
    def call(self, inputs, training=True):
        return self.__call__(inputs, training)

    def __call__(self, inputs, training=True):
        vae_z, a = inputs[:, :, :self.hps.output_seq_width], inputs[:, :, self.hps.output_seq_width:]

        z_a = tf.concat([vae_z, a], axis=2)

        rnn_out, _, _ = self.inference_base(z_a)

        rnn_out = tf.reshape(rnn_out, [-1, self.hps.rnn_size])
        out = self.out_net(rnn_out)
        return out

def rnn_next_state(rnn, z, a, prev_state):
    z = tf.cast(tf.reshape(z, [1, 1, -1]), tf.float32)
    a = tf.cast(tf.reshape(a, [1, 1, -1]), tf.float32)
    z_a = tf.concat([z, a], axis=2)
    _, h, c = rnn.inference_base(z_a, initial_state=prev_state)
    return [h, c]

def rnn_output_size(mode):
  if mode == MODE_ZCH:
    return (32+256+256)
  if (mode == MODE_ZC) or (mode == MODE_ZH):
    return (32+256)
  return 32 # MODE_Z or MODE_Z_HIDDEN

def rnn_init_state(rnn):
  return rnn.inference_base.cell.get_initial_state(batch_size=1, dtype=tf.float32) 

def rnn_output(state, z, mode):
  state_h, state_c = state[0], state[1]
  if mode == MODE_ZCH:
    return np.concatenate([z, np.concatenate((state_c,state_h), axis=1)[0]])
  if mode == MODE_ZC:
    return np.concatenate([z, state_c[0]])
  if mode == MODE_ZH:
    return np.concatenate([z, state_h[0]])
  return z # MODE_Z or MODE_Z_HIDDEN

