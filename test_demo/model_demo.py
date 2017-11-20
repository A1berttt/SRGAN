# modify name parameter in SRGAN_g model

import tensorlayer as tl
from tensorlayer.layers import *

def SRGAN_g(t_image, idx, is_train = False, reuse = False):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name=('in' + str(idx)))
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n