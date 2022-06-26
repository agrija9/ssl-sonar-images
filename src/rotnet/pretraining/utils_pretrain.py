import numpy as np
import tensorflow as tf
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pdb

def map_decorator(func):
    def wrapper(*args):
        return tf.py_function(
            func=func,
            inp=[*args],
            Tout=[a.dtype for a in args])
    return wrapper

@map_decorator
def apply_random_shift(image, label):
    """
    Page 4 pretrained models sonar images.
    """
    image = image.numpy() # (w, h, 1) accepted by tf function
    image_width = image.shape[0]
    image_height = image.shape[1]
    shift_w = np.random.uniform(0, 0.008*image_width)
    shift_h = np.random.uniform(0, 0.008*image_height)
    shifted_image = tf.keras.preprocessing.image.random_shift(image, shift_w, shift_h, row_axis=1, col_axis=1, channel_axis=0,
                                                              fill_mode='nearest', cval=0.0, interpolation_order=1)

    # shifted_image = shifted_image[:,:,:1] # (w, h, 1) dim needed for model
    # print(shifted_image.shape)
    # plt.imshow(shifted_image[:,:,0])
    # plt.show()
    # pdb.set_trace()

    return shifted_image, label

@map_decorator
def apply_random_flip(image, label):
    """
    """
    # data_flip = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical', seed=0.5)])
    image = image.numpy()
    flipped_image = tf.image.random_flip_left_right(image, seed=5)

    return flipped_image, label
