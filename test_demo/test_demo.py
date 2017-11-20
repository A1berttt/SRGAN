# a demo for test module of srgan by lee

from config import config, log_config
import tensorflow as tf
import tensorlayer as tl
from main import read_all_imgs
import time
import scipy
from tensorlayer.layers import *

def SRGAN_g(t_image, idx, is_train = False, reuse = False):
    idx = str(idx)
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("SRGAN_g", reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(t_image, name=('in' + str(idx)))
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c/' + idx)
        temp = n

        # B residual blocks
        for i in range(16):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name=('n64s1/c1/%s/' + idx) % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name=('n64s1/b1/%s/' + idx) % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name=('n64s1/c2/%s/' + idx) % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name=('n64s1/b2/%s/' + idx) % i)
            nn = ElementwiseLayer([n, nn], tf.add, ('b_residual_add/%s/' + idx) % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name=('n64s1/c/m/' + idx))
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m/' + idx)
        n = ElementwiseLayer([n, temp], tf.add, 'add3/' + idx)
        # B residual blacks end

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1/' + idx)
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1/' + idx)

        n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2/' + idx)
        n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2/' + idx)

        n = Conv2d(n, 3, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out/' + idx)
        return n

def session_1():
    log_config('./log_file.txt', config)

# generate hr_image of all test_lr_image by lee
def session_2():
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    valid_lr_imgs = read_all_imgs(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    valid_hr_imgs = read_all_imgs(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)

    n_of_images = len(valid_lr_img_list)

    for i in range(n_of_images):
        valid_hr_img = valid_hr_imgs[i]
        valid_lr_img = valid_lr_imgs[i]
        valid_lr_img = (valid_lr_img / 127.5) - 1
        size = valid_lr_img.shape
        t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]])
        net_g = SRGAN_g(t_image, i, is_train=False, reuse=False)

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess, name='../checkpoint' + '/g_srgan.npz', network=net_g)

        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})

        print("took: %4.4fs" % (time.time() - start_time))
        print("LR size: %s /  generated HR size: %s" % (size, out.shape))
        print("[*] save images")

        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, 'evaluate_all/' + 'valid_bicubic' + str(i) + '.png')
        tl.vis.save_image(out[0], 'evaluate_all/' + 'valid_gen_' + str(i) + '.png')
        tl.vis.save_image(valid_hr_img, 'evaluate_all/' + 'valid_hr' + str(i) + '.png')

def main():
    # session_1()
    session_2()


if __name__ == '__main__':
    main()
