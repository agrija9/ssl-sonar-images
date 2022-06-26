import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import os
import h5py
import random

class CIFAR10(object):
    """
    This class defines the CIFAR-10 dataset. It reads the pickled files as
    provided by the dataset's official website.
    """
    def __init__(self, data_dir, height, width):
        if data_dir[-1] != "/":
            data_dir = data_dir + "/"
        self.data_dir = data_dir
        self.height = height
        self.width = width

    def get_training_data(self):
        """
        Returns training images and labels.
        These  helper functions are optimized for pickled CIFAR-10.
        """
        print("[INFO] Getting Training Data")
        images = []
        labels = []
        for i in range(1,6):
            images.append(self._get_next_batch_from_file(i)[b'data'])
            labels.append(self._get_next_batch_from_file(i)[b'labels'])
        return self.convert_images(np.concatenate(images)), np.concatenate(labels)

    def _get_next_batch_from_file(self, batch_number):
        data = self._unpickle_data(self.data_dir + self._get_batch_name(batch_number))
        return data

    def _unpickle_data(self, filepath):
        with open(filepath, 'rb') as data:
            dict = pickle.load(data, encoding='bytes')
        return dict

    def _get_batch_name(self, number):
        return "data_batch_{0}".format(number)

    def convert_images(self, raw_images):
        """
        This function normalizes the input images and converts them to
        the appropriate shape: batch_size x height x width x channels
        """
        images = raw_images / 255.0
        images = raw_images.reshape([-1, 3, self.height, self.width])
        images = images.transpose([0, 2, 3, 1])
        return images

    def get_test_data(self):
        """
        Write a function to get the test set from disk.
        """
        d = self._unpickle_data(self.data_dir + 'test_batch')
        images = self.convert_images(d[b'data'])
        return images, d[b'labels']

    def preprocess(self, images):
        """
        Rotates images and stores labels for each rotation.
        0: 0 degrees
        1: 90 degrees
        2: 180 degrees
        3: 270 degrees
        This is the core component of RotNet.
        """
        images_rotated = []
        labels_rotated = []
        for num_r in [0, 1, 2, 3]:
            for i in range(len(images)):
                r_im = np.rot90(images[i], k=num_r).reshape(1, 32, 32, 3)
                images_rotated.append(r_im)
                labels_rotated.append(num_r)
        return np.concatenate(images_rotated), np.array(labels_rotated)

    @staticmethod
    def print_image_to_screen(data):
        """
        """
        img = Image.fromarray(data, 'RGB')
        img.show()


class SONAR(object):

    # NOTE: this is deprecated, it reads from raw images,
    # currently reading from hdf5

    def __init__(self, data_dir):
        """
        """
        self.data_dir = data_dir

    def _get_single_sonar_image(self, file):
        """
        """
        sonar_image = tf.io.read_file(file)
        sonar_image = tf.image.decode_png(sonar_image)
        sonar_image = tf.image.resize(sonar_image, (96, 96), method="bilinear") # resizing all sonar images to 96x96
        sonar_image = sonar_image[:,:,:1].numpy() # return image w 1 channel (grayscale)
        return sonar_image

    def _normalize_image(self, image):
        """
        """
        # check if needed (used in CIFAR)
        # test = sample.reshape([-1, 1, 63, 63])
        # test = test.transpose([0, 2, 3, 1])
        return image / 255.0

    def get_sonar_data(self):
        """
        """
        sonar_images = []
        for subdir, _, files in os.walk(self.data_dir):
            for file in files:
                sonar_image = self._get_single_sonar_image(os.path.join(subdir, file))
                normalized_sonar_image = self._normalize_image(sonar_image)
                sonar_images.append(normalized_sonar_image)
        # check if ok concatenating varying 2d array dims
        # sonar_images = np.asarray(sonar_images, dtype="object")
        return sonar_images

    def generate_rotations(self, images):
        """
        Rotates images and stores labels for each rotation.
        0: 0 degrees
        1: 90 degrees
        2: 180 degrees
        3: 270 degrees
        This is the core component of RotNet.
        """
        images_rotated = []
        labels_rotated = []
        for num_r in [0, 1, 2, 3]:
            for i in range(len(images)):
                # r_im = np.rot90(images[i], k=num_r).reshape(1, 32, 32, 3)
                r_im = np.rot90(images[i], k=num_r)
                images_rotated.append(r_im)
                labels_rotated.append(num_r)
        # return np.concatenate(images_rotated), np.array(labels_rotated)
        # return np.asarray(images_rotated, dtype="object"), np.array(labels_rotated)
        return images_rotated, np.array(labels_rotated)

    def plot_sonar_image(image):
        """
        """
        plt.imshow(image[:,:,0], cmap="gray")


class SonarDebrisSelfSupervised(object):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _normalize_images(self, images):

        """
        """
        # test = sample.reshape([-1, 1, 63, 63])
        # test = test.transpose([0, 2, 3, 1])
        return [element/255.0 for element in images]

    def get_sonar_data(self):

        """
        Reads from HDF5 file containing sonar data.
        Returns list of np arrays.
        Resized to 96x96 dimensions.
        """
        with h5py.File(self.data_dir, "r") as f:
            # list all groups
            print("hdf5 dataset keys: %s" % f.keys())
            # labels_key = list(f.keys())[0] # not using labels for SSL methods

            # get the data
            # labels = list(f[labels_key])
            x_train = f["x_train"][...].astype(np.float32)
            x_val = f["x_val"][...].astype(np.float32)
            x_test = f["x_test"][...].astype(np.float32)

            print("[INFO] Original data dimensions")
            print("Train", len(x_train))
            print("Val", len(x_val))
            print("Test", len(x_test))

            # normalize data
            # x_train = self._normalize_images(x_train)
            # x_val = self._normalize_images(x_val)
            # x_test = self._normalize_images(x_test)

            # matias normalization
            # multiply by 255 because hdf5 file comes normalized
            x_train *= 255.0
            x_val *= 255.0
            x_test *= 255.0

            # substract mean (this dataset is not divided by 255)
            x_train -= 84.51
            x_val -= 84.51
            x_test  -= 84.51

        return x_train, x_val, x_test

    def generate_rotations(self, images):
        """
        Rotates images and stores labels for each rotation.
        0: 0 degrees
        1: 90 degrees
        2: 180 degrees
        3: 270 degrees
        This is the core component of RotNet.
        """
        images_rotated = []
        labels_rotated = []
        for num_r in [0, 1, 2, 3]:
            for i in range(len(images)):
                r_im = np.rot90(images[i], k=num_r)
                images_rotated.append(r_im)
                labels_rotated.append(num_r)

        # return np.concatenate(images_rotated), np.array(labels_rotated)
        return np.asarray(images_rotated), np.array(labels_rotated)

class SonarWildDataSelfSupervised(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def get_sonar_data(self):
        """
        Reads from HDF5 file containing sonar data (resized to fix dims).
        Returns list of np arrays containing image data.
        """
        random.seed(10)

        print("[INFO] Retrieving Wild Sonar Data")
        with h5py.File(self.file_path, "r") as f:
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            print(f[a_group_key])

            # Get wild data # NOTE: casting to list takes way too long
            data = np.array(f[a_group_key]).astype(np.float32)

            # generate train/val/test sets
            # NOTE: downsampling dataset by half (63K/2) due to memory issues
            x_train = data[:int(len(data)*0.7),:]
            x_val = data[int(len(data)*0.7):int(len(data)*0.7)+int(len(data)*0.15),:]
            x_test = data[int(len(data)*0.7)+int(len(data)*0.15):,:]

            print("[INFO] Original Data dimensions")
            print("Train", len(x_train))
            print("Val", len(x_val))
            print("Test", len(x_test))
            print()

            # 1/2 too big, testing 1/10
            x_train = x_train[:int(len(x_train)/10)]
            x_val = x_val[:int(len(x_val)/10)]
            x_test = x_test[:int(len(x_test)/45)]

            print("[INFO] Downsampled Data dimensions")
            print("Train", len(x_train))
            print("Val", len(x_val))
            print("Test", len(x_test))

        return x_train, x_val, x_test
        # return (x_train, _), (x_val, _), (x_test, _)

    def generate_rotations(self, images):
        """
        Rotates images and stores labels for each rotation.
        0: 0 degrees
        1: 90 degrees
        2: 180 degrees
        3: 270 degrees
        This is the core component of RotNet.
        """
        images_rotated = []
        labels_rotated = []
        for num_r in [0, 1, 2, 3]:
            for i in range(len(images)):
                r_im = np.rot90(images[i], k=num_r)
                images_rotated.append(r_im)
                labels_rotated.append(num_r)

        # return np.concatenate(images_rotated), np.array(labels_rotated)
        return np.asarray(images_rotated), np.array(labels_rotated)

class SonarDebrisSupervised(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def _normalize_images(self, images):

        """
        """
        # HDF5 images are already normalized
        return [element/255.0 for element in images]

    def get_sonar_data(self):

        """
        Reads from HDF5 file containing sonar data (resized to fix dims).
        Returns list of np arrays containing image data.
        """

        print("[INFO] Retrieving Sonar Debris Supervised Data")
        with h5py.File(self.file_path, "r") as f:
            # get images and labels
            x_train = f["x_train"][...].astype(np.float32)
            y_train = f["y_train"][...]

            x_val = f["x_val"][...].astype(np.float32)
            y_val = f["y_val"][...]

            x_test = f["x_test"][...].astype(np.float32)
            y_test = f["y_test"][...]

            # matias normalization
            # multiply by 255 because hdf5 file comes as 1/255
            x_train *= 255.0
            x_val *= 255.0
            x_test *= 255.0

            # substract mean (this dataset is not divided by 255)
            x_train -= 84.51
            x_val -= 84.51
            x_test  -= 84.51

            print("[INFO] Data dimensions")
            print("Train", len(x_train))
            print("Val", len(x_val))
            print("Test", len(x_test))

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class SonarTurnedTableSupervised(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def _normalize_images(self, images):
        """
        Normalize sonar images by 1/255.
        """
        return [element/255.0 for element in images]

    def get_sonar_data(self):
        """
        Reads from HDF5 file containing sonar data (resized to fix dims).
        Returns list of np arrays containing image data.
        """

        print("[INFO] Retrieving Sonar Turned Table Supervised Data")

        with h5py.File(self.file_path, "r") as f:
            # list all groups
            print("hdf5 dataset keys: %s" % f.keys())

            # get images and labels
            # x_train_val = list(f["x_train"])
            # y_train_val = list(f["y_train"])
            x_train = f["x_train"][...].astype(np.float32)
            y_train = f["y_train"][...]

            # x_test = list(f["x_test"])
            # y_test = list(f["y_test"])
            x_test = f["x_test"][...].astype(np.float32)
            y_test = f["y_test"][...]

            _, x_val, _, y_val = train_test_split(x_test, y_test, train_size=0.5)

            print("[INFO] Data dimensions")
            print("Train", len(x_train))
            print("Val", len(x_val))
            print("Test", len(x_test))

            # matias normalization
            # multiply by 255 because hdf5 file comes as 1/255
            x_train *= 255.0
            x_val *= 255.0
            x_test *= 255.0

            x_train -= 84.51
            x_val -= 84.51
            x_test  -= 84.51

        # x_train = self._normalize_images(x_train)
        # x_val = self._normalize_images(x_val)
        # x_test = self._normalize_images(x_test)

        # return (x_train_val, y_train_val), (x_val, y_val), (x_test, y_test)
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
