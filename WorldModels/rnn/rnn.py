import numpy as np
from collections import namedtuple
import json
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4

@tf.function
def sample_vae(vae_mu, vae_logvar):
    sz = vae_mu.shape[1]
    mu_logvar = tf.concat([vae_mu, vae_logvar], axis=1)
    z = tfp.layers.DistributionLambda(lambda theta: tfp.distributions.MultivariateNormalDiag(loc=theta[:, :sz], scale_diag=tf.exp(theta[:, sz:])), dtype=tf.float16)
    return z(mu_logvar)

class MDNRNN(tf.keras.Model):
    def __init__(self, args):
        super(MDNRNN, self).__init__()
        self.args = args
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.rnn_learning_rate, clipvalue=self.args.rnn_grad_clip)

        self.loss_fn = self.get_loss() 

        lstm_cell = tf.keras.layers.LSTMCell(units=args.rnn_size)

        self.inference_base = tf.keras.layers.RNN(cell=lstm_cell, return_sequences=True, return_state=True, time_major=False)

        self.out_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.args.rnn_size),
            tf.keras.layers.Dense(args.rnn_out_size, name="mu_logstd_logmix_net")])

        super(MDNRNN, self).build((self.args.rnn_batch_size, self.args.rnn_max_seq_len, self.args.rnn_input_seq_width))

    def get_loss(self):
        num_mixture = self.args.rnn_num_mixture
        batch_size = self.args.rnn_batch_size
        z_size = self.args.z_size
        
        """Construct a loss functions for the MDN layer parametrised by number of mixtures."""
        # Construct a loss function with the right number of mixtures and outputs
        def z_loss_func(y_true, y_pred):
            '''
            This loss function is defined for N*k components each containing a gaussian of 1 feature
            '''
            mdnrnn_params = y_pred

            y_true = tf.reshape(y_true, [batch_size, -1, z_size + 1]) # +1 for mask
            z_true, mask = y_true[:, :, :-1], y_true[:, :, -1:]

            # Reshape inputs in case this is used in a TimeDistribued layer
            mdnrnn_params = tf.reshape(mdnrnn_params, [-1, 3*num_mixture], name='reshape_ypreds')
            vae_z, mask = tf.reshape(z_true, [-1, 1]), tf.reshape(mask, [-1, 1])
            
            out_mu, out_logstd, out_logpi = tf.split(mdnrnn_params, num_or_size_splits=3, axis=1, name='mdn_coef_split')
            out_logpi = out_logpi - tf.reduce_logsumexp(input_tensor=out_logpi, axis=1, keepdims=True) # normalize

            logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))
            lognormal = -0.5 * ((vae_z - out_mu) / tf.exp(out_logstd)) ** 2 - out_logstd - logSqrtTwoPI
            v = out_logpi + lognormal
            
            z_loss = -tf.reduce_logsumexp(input_tensor=v, axis=1, keepdims=True)
            mask = tf.reshape(tf.tile(mask, [1, z_size]), [-1, 1]) # tile b/c we consider z_loss is flattene
            z_loss = mask * z_loss # don't train if episode ends
            z_loss = tf.reduce_sum(z_loss) / tf.reduce_sum(mask) 
            return z_loss

        def d_loss_func(y_true, y_pred):
            d_pred = y_pred

            y_true = tf.reshape(y_true, [batch_size, -1, 1 + 1]) # b/c tf is stupid
            d_true, mask = y_true[:, :, :-1], y_true[:, :, -1:]
            d_true, mask = tf.reshape(d_true, [-1, 1]), tf.reshape(mask, [-1, 1])
            
            d_loss = tf.expand_dims(tf.keras.losses.binary_crossentropy(y_true=d_true, y_pred=d_pred, from_logits=True), axis=-1)
            d_loss = mask * d_loss
            d_loss = tf.reduce_sum(d_loss) / tf.reduce_sum(mask) # mean of unmasked 
            return d_loss

        if self.args.env_name == 'DoomTakeCover-v0':
            return z_loss_func, d_loss_func
        else:
            return z_loss_func

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
        rnn_out, _, _ = self.inference_base(inputs)

        rnn_out = tf.reshape(rnn_out, [-1, self.args.rnn_size])
        out = self.out_net(rnn_out)
        if self.args.env_name == 'CarRacing-v0':
          return out
        else: 
          mdnrnn_params, done_logits = out[:, :-1], out[:, -1:]
          return mdnrnn_params, done_logits

@tf.function
def rnn_next_state(rnn, z, a, prev_state):
    z = tf.cast(tf.reshape(z, [1, 1, -1]), tf.float32)
    a = tf.cast(tf.reshape(a, [1, 1, -1]), tf.float32)
    z_a = tf.concat([z, a], axis=2)
    _, h, c = rnn.inference_base(z_a, initial_state=prev_state)
    return [h, c]

@tf.function
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

@tf.function
def rnn_sim(rnn, z, states, a):
  if rnn.args.env_name == 'CarRacing-v0':
    raise ValueError('Not implemented yet for CarRacing')
  z = tf.reshape(tf.cast(z, dtype=tf.float32), (1, 1, rnn.args.z_size))
  a = tf.reshape(tf.cast(a, dtype=tf.float32), (1, 1, rnn.args.a_width))
  input_x = tf.concat((z, a), axis=2)
  rnn_out, h, c = rnn.inference_base(input_x, initial_state=states)
  rnn_state = [h, c]
  rnn_out = tf.reshape(rnn_out, [-1, rnn.args.rnn_size])
  out = rnn.out_net(rnn_out)

  mdnrnn_params, d_logits = out[:, :-1], out[:, -1:]
  mdnrnn_params = tf.reshape(mdnrnn_params, [-1, 3*rnn.args.rnn_num_mixture])

  mu, logstd, logpi = tf.split(mdnrnn_params, num_or_size_splits=3, axis=1)
  logpi = logpi - tf.reduce_logsumexp(input_tensor=logpi, axis=1, keepdims=True) # normalize

  d_dist = tfd.Binomial(total_count=1, logits=d_logits)
  d = tf.squeeze(d_dist.sample()) == 1.0

  cat = tfd.Categorical(logits=logpi)
  component_splits = [1] * rnn.args.rnn_num_mixture
  mus = tf.split(mu, num_or_size_splits=component_splits, axis=1)
  sigs = tf.split(tf.exp(logstd), num_or_size_splits=component_splits, axis=1)
  coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale in zip(mus, sigs)]
  mixture = tfd.Mixture(cat=cat, components=coll)

  z = tf.reshape(mixture.sample(), shape=(-1, rnn.args.z_size))

  r = 1.0 # For Doom Reward is always 1.0 if the agent is alive

  return rnn_state, z, r, d
