import config
import tensorlayer as tl
from utils import *
from skimage.measure import compare_mse,compare_psnr,compare_ssim

def get_compare_images():
    gen_image_list = sorted(tl.files.load_file_list(path=config.COMPARE.img_path, regx='valid_gen*.png', printable=False))
    # bic_image_list = sorted(tl.files.load_file_list(path=config.COMPARE.img_path, regx='valid_bicubic*.png', printable=False))
    hr_image_list = sorted(tl.files.load_file_list(path=config.COMPARE.img_path, regx='valid_hr*.png', printable=False))

    def read_all_imgs(img_list, path='', n_threads = 32):
        imgs = []
        for idx in range(0, len(img_list), n_threads):
            b_imgs_list = img_list[idx: idx + n_threads]
            b_imgs = tl.prepro.threading_data(b_imgs_list, fn = get_imgs_fn, path = path)
            imgs.extend(b_imgs)
            print('read %d from $s' % (len(imgs), path))
        return imgs

    gen_imgs = read_all_imgs(gen_image_list, path=config.COMPARE.img_path, n_threads=32)
    # bic_imgs = read_all_imgs(bic_image_list, path=config.COMPARE.img_path, n_threads=32)
    hr_imgs = read_all_imgs(hr_image_list, path=config.COMPARE.img_path, n_threads=32)

    return gen_imgs, hr_imgs

# fn = compare_mse, compare_psnr or compare_ssim
def compare_imgs(sr, hr, fn=None):
    compare_value = 0.
    for s, h in zip(sr, hr):
        compare_value += fn(s, h)
    compare_value /= len(hr)
    return compare_value
