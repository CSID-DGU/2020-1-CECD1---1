import numpy as np
import tensorflow as tf

from scipy import interpolate
from .model import Model, default_opt

from .layers.subpixel import SubPixel1D, SubPixel1D_v2

from keras import backend as K
from keras.layers import merge, concatenate, add, Dense, Flatten, Reshape, Input
from keras.layers.core import Activation, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import normal, orthogonal

# ----------------------------------------------------------------------------

class AudioUNet(Model):
  """Generic tensorflow model training code"""

  def __init__(self, from_ckpt=False, n_dim=None, r=2,
               opt_params=default_opt, log_prefix='./run'):
    # perform the usual initialization
    self.r = r
    Model.__init__(self, from_ckpt=from_ckpt, n_dim=n_dim, r=r,
                   opt_params=opt_params, log_prefix=log_prefix)

  def create_model(self, n_dim, r, reuse=False):
    # load inputs
    X, _, _, _ = self.inputs
    K.set_session(self.sess)

    with tf.name_scope('generator'):
      x1 = X
    #   x1 = Input(shape=(8192, 1))
      print('------------------------------')
      print(x1.get_shape())
      print('------------------------------')
      L = self.layers
      # dim/layer: 4096, 2048, 1024, 512, 256, 128,  64,  32,
      # n_filters = [  64,  128,  256, 384, 384, 384, 384, 384]
      n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
      # n_filters = [  256,  512,  512, 512, 512, 1024, 1024, 1024]
      # n_filtersizes = [129, 65,   33,  17,  9,  9,  9, 9]
      # n_filtersizes = [31, 31,   31,  31,  31,  31,  31, 31]
      n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
      downsampling_l = []

      print('building model...')
      
      x_down, downsampling_l = self.downsampling(x1, L, n_filters, n_filtersizes)
      x2, _ = self.downsampling(x1, L, n_filters, n_filtersizes, reuse=True)
      x2 = tf.stop_gradient(x2)
      x_global, x3 = self.global_feature_net(x2)

      x_bottle = self.bottleneck_layer(x_down, x_global, n_filters, n_filtersizes)
    #   x = self.bottleneck_layer_origin(x, n_filters, n_filtersizes)
      x_upsam = self.upsampling(x_bottle, L, n_filters, n_filtersizes, downsampling_l)
      x_fin = self.final_layer(x_upsam)

    #   g = add([x_fin, X])

    # print('--------------')
    # print(g)
    # print('--------------')

    # return x3, g
    return x3, x_fin

  def downsampling(self, x, L, n_filters, n_filtersizes, reuse=False):
    downsampling_l = []

    # downsampling layers
    for l, nf, fs in zip(range(L), n_filters, n_filtersizes):
      with tf.variable_scope('downsc_conv%d' % l, reuse=reuse):
        x = tf.layers.conv1d(inputs=x, filters=nf, kernel_size=fs, strides=2, padding='same', activation=None,
                kernel_initializer=tf.initializers.orthogonal)
        x = tf.nn.leaky_relu(x)
        print('D-Block: ', x.get_shape())
        downsampling_l.append(x)

    return x, downsampling_l
    
  def bottleneck_layer(self, x, x2, n_filters, n_filtersizes):
    # bottleneck layer
    with tf.name_scope('bottleneck_conv'):
      x_shape = x.get_shape()
      reshaped_x2 = x2[:, tf.newaxis, :]
      reshaped_x2 = tf.tile(reshaped_x2, [1, x_shape[1].value, 1])
      reshaped_x2 = tf.reshape(reshaped_x2, shape=[-1, x_shape[1].value, x2.get_shape()[-1].value])
      print('Reshaped_x2: ', reshaped_x2.get_shape())

      x = concatenate([x, reshaped_x2])
      print('Concatenate: ', x.get_shape())
      x = tf.layers.conv1d(inputs=x, filters=n_filters[-1], kernel_size=n_filtersizes[-1],
                strides=2, padding='same', activation=None, kernel_initializer=tf.initializers.orthogonal)
      x = tf.nn.dropout(x, 0.5)
      # x = BatchNormalization(mode=2)(x)
      x = tf.nn.leaky_relu(x)
      print('bottleneck: ', x.get_shape())

      return x

  def bottleneck_layer_origin(self, x, n_filters, n_filtersizes):
    # bottleneck layer
    with tf.name_scope('bottleneck_conv'):
      x = tf.layers.conv1d(inputs=x, filters=n_filters[-1], kernel_size=n_filtersizes[-1],
                strides=2, padding='same', activation=None, kernel_initializer=tf.initializers.orthogonal)
      x = tf.nn.dropout(x, 0.5)
      # x = BatchNormalization(mode=2)(x)
      x = tf.nn.leaky_relu(x)

      return x

  def upsampling(self, x, L, n_filters, n_filtersizes, downsampling_l):
    # upsampling layers
      layers = list(zip(range(L), n_filters, n_filtersizes, downsampling_l))
      layers.reverse()
      for l, nf, fs, l_in in layers:
        with tf.name_scope('upsc_conv%d' % l):
          # (-1, n/2, 2f)
          x = tf.layers.conv1d(inputs=x, filters=2*nf, kernel_size=fs, padding='same', activation=None,
                kernel_initializer=tf.initializers.orthogonal)
          # x = BatchNormalization(mode=2)(x)
          x = tf.nn.dropout(x, 0.5)
          x = tf.nn.relu(x)
          # (-1, n, f)
          x = SubPixel1D(x, r=2) 
          # (-1, n, 2f)
          x = concatenate([x, l_in])
          print('U-Block: ', x.get_shape())

      return x

  def global_feature_net(self, x):
    with tf.name_scope('global_feature_net'):
      x = tf.layers.conv1d(inputs=x, filters=256, kernel_size=3, strides=1, padding='valid', activation=None,
                  kernel_initializer=tf.initializers.orthogonal)
      x = tf.nn.dropout(x, 0.5)
      x = tf.nn.relu(x)
      x = tf.layers.conv1d(inputs=x, filters=128, kernel_size=3, strides=1, padding='valid', activation=None,
                  kernel_initializer=tf.initializers.orthogonal)
      x = tf.nn.dropout(x, 0.5)
      x = tf.nn.relu(x)
      x = tf.layers.conv1d(inputs=x, filters=64, kernel_size=3, strides=1, padding='valid', activation=None,
                  kernel_initializer=tf.initializers.orthogonal)
      x = tf.nn.relu(x)


      flattened = tf.contrib.layers.flatten(x)
      print('Flattened: ', flattened.get_shape())
      fc0 = tf.contrib.layers.fully_connected(flattened, num_outputs=512,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu
													)
      fc0_1 = tf.contrib.layers.fully_connected(fc0, num_outputs=256,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=tf.nn.relu
													)
      fc1 = tf.contrib.layers.fully_connected(fc0_1, num_outputs=128,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=None
													)
      fc1 = tf.nn.relu(fc1)
      fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=64,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=None
													)
      fc2_1 = tf.nn.sigmoid(fc2)
      fc3 = tf.contrib.layers.fully_connected(fc2_1, num_outputs=10,
													weights_initializer=tf.contrib.layers.xavier_initializer(),
													activation_fn=None
													)
      print('Global-Block: ', fc3.get_shape())

      return fc1, fc3

  def final_layer(self, x):
    # final conv layer
    with tf.name_scope('lastconv'):
      x = tf.layers.conv1d(inputs=x, filters=2, kernel_size=9, padding='same', activation=None,
                kernel_initializer=tf.initializers.orthogonal)
    #   x = Conv1D(2, 9, 
    #         activation=None, padding='same', kernel_initializer=normal_init)(x)    
      x = SubPixel1D(x, r=2) 
      print(x.get_shape())

      return x

  def predict(self, X):
    assert len(X) == 1
    x_sp = spline_up(X, self.r)
    x_sp = x_sp[:len(x_sp) - (len(x_sp) % (2**(self.layers+1)))]
    X = x_sp.reshape((1,len(x_sp),1))
    feed_dict = self.load_batch((X,X,[[0]]), train=False)
    return self.sess.run(self.predictions, feed_dict=feed_dict)

# ----------------------------------------------------------------------------
# helpers

def normal_init(shape, dim_ordering='tf', name=None):
  # return normal(shape, scale=1e-3, name=name, dim_ordering=dim_ordering)
  return normal()(shape)

def orthogonal_init(shape, dim_ordering='tf', name=None):
  # return orthogonal(shape, name=name, dim_ordering=dim_ordering)
  return orthogonal()(shape)

def spline_up(x_lr, r):
  x_lr = x_lr.flatten()
  x_hr_len = len(x_lr) * r
  x_sp = np.zeros(x_hr_len)
  
  i_lr = np.arange(x_hr_len, step=r)
  i_hr = np.arange(x_hr_len)
  
  f = interpolate.splrep(i_lr, x_lr)

  x_sp = interpolate.splev(i_hr, f)

  return x_sp