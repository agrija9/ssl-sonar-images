import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import pdb
import numpy as np
from datasets import (CIFAR10,
                      SONAR,
                      SonarDebrisSelfSupervised,
                      SonarDebrisSupervised,
                      SonarTurnedTableSupervised,
                      SonarWildDataSelfSupervised)

from utils_pretrain import apply_random_shift, apply_random_flip

def load_CIFAR10(data_dir, batch_size):
    """
    """

    print("[INFO] Loading CIFAR-10 dataset")

    image_height = 32
    image_width = 32

    # Define dataset object
    dataset_object = CIFAR10(data_dir, image_height, image_width)

    # Obtain train/validation data and one-hot encode labels
    images, labels = dataset_object.get_training_data() # original labels (1-10)
    images, labels = dataset_object.preprocess(images) # rotation labels (1-4)

    images_train, images_val, labels_train, labels_val = train_test_split(images, labels)
    labels_train = tf.keras.utils.to_categorical(labels_train)
    labels_val = tf.keras.utils.to_categorical(labels_val)

    # Obtain test data
    images_test, labels_test = dataset_object.get_test_data()
    images_test, labels_test = dataset_object.preprocess(images_test)
    labels_test = tf.keras.utils.to_categorical(labels_test)

    print("[INFO] Numpy data dimensions")
    print(images_train.shape) # from 50,000 to 50,000 * 4
    print(images_val.shape)
    print(images_test.shape)
    # pdb.set_trace()

    # Create Tensorflow Datasets by consuming numpy arrays
    # Train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(images_train)).batch(batch_size)
    train_dataset = train_dataset.prefetch(25)

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((images_val, labels_val))
    val_dataset = val_dataset.shuffle(buffer_size=len(images_val)).batch(batch_size)
    val_dataset = val_dataset.prefetch(25)

    # Test dataset
    test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
    test_dataset = test_dataset.shuffle(buffer_size=len(images_test)).batch(batch_size)
    test_dataset = test_dataset.prefetch(25)

    print("[INFO] Tensorflow data dimensions")
    print(train_dataset)
    print(val_dataset)
    print(test_dataset)
    # pdb.set_trace()

    return train_dataset, val_dataset, test_dataset

def load_SONAR(data_dir, batch_size):
    """
    """
    # NOTE: this is deprecated, it reads from raw images,
    # currently reading from hdf5

    print("[INFO] Loading SONAR dataset")

    dataset_object = SONAR(data_dir)

    original_images = dataset_object.get_sonar_data()
    images, labels = dataset_object.generate_rotations(original_images) # should be 9384 but it is 9456

    images_train, images_val, labels_train, labels_val = train_test_split(images, labels)
    labels_train = tf.keras.utils.to_categorical(labels_train)
    labels_val = tf.keras.utils.to_categorical(labels_val)

    print("[INFO] Numpy data dimensions")
    print(np.shape(images_train))
    print(np.shape(images_val))

    # Create Tf datasets from generators (due to varying image sizes)
    # Train data
    # train_dataset = tf.data.Dataset.from_generator(lambda: (images_train, labels_train),
    #                                                output_types = (tf.uint8, tf.float32),
    #                                                output_shapes = (tf.TensorShape([None, None, 1]), tf.TensorShape([4])))

    train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(images_train)).batch(batch_size)
    train_dataset = train_dataset.prefetch(25)

    # Validation data
    # val_dataset = tf.data.Dataset.from_generator(lambda: (images_val, labels_val),
    #                                              output_types = (tf.uint8, tf.float32),
    #                                              output_shapes = (tf.TensorShape([None, None, 1]), tf.TensorShape([4])))

    val_dataset = tf.data.Dataset.from_tensor_slices((images_val, labels_val))
    val_dataset = val_dataset.shuffle(buffer_size=len(images_val)).batch(batch_size)
    val_dataset = val_dataset.prefetch(25)

    print("[INFO] Tensorflow data dimensions")
    print(train_dataset)
    print(val_dataset)
    # pdb.set_trace()

    return train_dataset, val_dataset

def load_sonar_debris_self_supervised(data_dir, batch_size):
    """
    """
    dataset_object = SonarDebrisSelfSupervised(data_dir)

    # Read data, generate rotations, convert to categorical
    x_train, x_val, x_test = dataset_object.get_sonar_data()
    images_train, labels_train = dataset_object.generate_rotations(x_train)
    images_val, labels_val = dataset_object.generate_rotations(x_val)
    images_test, labels_test = dataset_object.generate_rotations(x_test)

    labels_train = tf.keras.utils.to_categorical(labels_train)
    labels_val = tf.keras.utils.to_categorical(labels_val)
    labels_test = tf.keras.utils.to_categorical(labels_test)

    print("[INFO] Data dimensions after rotations")
    print(len(images_train))
    print(len(images_val))
    print(len(images_test))

    # Create Tf datasets
    # Train data
    # train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    # train_dataset = train_dataset.map(apply_random_shift)
    # train_dataset = train_dataset.shuffle(buffer_size=len(images_train)).batch(batch_size)
    # train_dataset = train_dataset.prefetch(25)

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    train_gen = image_gen.flow(images_train, labels_train, batch_size=batch_size, shuffle=True)
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                   output_types = (tf.float32, tf.float32))
                                                   # output_shapes = ([None, 96, 96, 1], [None, 4]))


    # Validation data
    val_dataset = tf.data.Dataset.from_tensor_slices((images_val, labels_val))
    val_dataset = val_dataset.shuffle(buffer_size=len(images_val)).batch(batch_size)
    val_dataset = val_dataset.prefetch(25)

    # Test data
    test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
    test_dataset = test_dataset.shuffle(buffer_size=len(images_test)).batch(len(images_test)) # feed full test set
    test_dataset = test_dataset.prefetch(25)

    print("[INFO] Tensorflow data dimensions")
    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    return train_dataset, val_dataset, test_dataset

def load_wild_sonar_data(file_path, batch_size):
    print()
    print("[INFO] Loading Tf datasets")

    dataset_object = SonarWildDataSelfSupervised(file_path)

    # Read data
    x_train, x_val, x_test = dataset_object.get_sonar_data()
    # (x_train, _), (x_val, _), (x_test, _) = dataset_object.get_sonar_data()
    images_train, labels_train = dataset_object.generate_rotations(x_train)
    images_val, labels_val = dataset_object.generate_rotations(x_val)
    images_test, labels_test = dataset_object.generate_rotations(x_test)

    labels_train = tf.keras.utils.to_categorical(labels_train)
    labels_val = tf.keras.utils.to_categorical(labels_val)
    labels_test = tf.keras.utils.to_categorical(labels_test)

    print("[INFO] Data dimensions after rotations")
    print(len(images_train))
    print(len(images_val))
    print(len(images_test))

    # Create TF Datasets

    # Train data
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1)
    train_gen = image_gen.flow(images_train, labels_train, batch_size=batch_size, shuffle=True)
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                   output_types = (tf.float32, tf.float32))
                                                   # output_shapes = ([None, 96, 96, 1], [None, 4]))

    # Validation data
    val_dataset = tf.data.Dataset.from_tensor_slices((images_val, labels_val))
    val_dataset = val_dataset.shuffle(buffer_size=len(images_val)).batch(batch_size)
    val_dataset = val_dataset.prefetch(25)

    # Test data
    test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
    test_dataset = test_dataset.shuffle(buffer_size=len(images_test)).batch(len(images_test)) # feed full test set
    test_dataset = test_dataset.prefetch(25)

    print("[INFO] Tensorflow data dimensions")
    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    return train_dataset, val_dataset, test_dataset
    # return (x_train, _), (x_val, _), (x_test, _)

def load_sonar_debris_supervised(file_path, batch_size):
    """
    Tensorflow data pipeline: loads data as numpy arrays, defines tf dataset, performs
    image augmentation (random shift and random flipping), returns model-ready data.
    """

    print()
    print("[INFO] Loading Tf datasets")

    dataset_object = SonarDebrisSupervised(file_path)

    # Read data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset_object.get_sonar_data()

    # Convert labels to categorical
    labels_train = tf.keras.utils.to_categorical(y_train)
    labels_val = tf.keras.utils.to_categorical(y_val)
    labels_test = tf.keras.utils.to_categorical(y_test)

    # Train data: apply real-time image augmentation using ImageDataGenerator
    # https://stackoverflow.com/questions/54606302/tf-data-dataset-from-tf-keras-preprocessing-image-imagedatagenerator-flow-from-d
    # https://stackoverflow.com/questions/56232389/why-imagedatagenerator-is-iterating-forever/56232612
    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                                height_shift_range=0.1,
                                                                horizontal_flip=True,
                                                                vertical_flip=True)

    train_gen = image_gen.flow(x_train, labels_train, batch_size=batch_size, shuffle=True)
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                   output_types = (tf.float32, tf.float32))
                                                   # output_shapes = ([None, 96, 96, 1], [None, 11]))

    # train_dataset = tf.data.Dataset.from_tensor_slices((x_train, labels_train))
    # train_dataset = train_dataset.map(apply_random_shift)
    # train_dataset = train_dataset.map(apply_random_flip)
    # train_dataset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)
    # train_dataset = train_dataset.prefetch(25)

    # Validation data
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, labels_val))
    val_dataset = val_dataset.shuffle(buffer_size=len(x_val)).batch(batch_size)
    val_dataset = val_dataset.prefetch(25)

    # Test data
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, labels_test))
    test_dataset = test_dataset.shuffle(buffer_size=len(x_test)).batch(len(x_test))
    test_dataset = test_dataset.prefetch(25)

    print()
    print("[INFO] Tensorflow data dimensions")
    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    # print()
    # print(next(iter(train_dataset)))
    # print(list(train_dataset.as_numpy_iterator()))
    # print(len(list(train_dataset.as_numpy_iterator())))
    # pdb.set_trace()

    return train_dataset, val_dataset, test_dataset

def load_sonar_turnedtable_supervised(file_path, batch_size):
    """
    """
    print()
    print("[INFO] Loading Tf datasets")

    dataset_object = SonarTurnedTableSupervised(file_path)

    # Read data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset_object.get_sonar_data()

    # Convert to categorical
    labels_train = tf.keras.utils.to_categorical(y_train)
    labels_val = tf.keras.utils.to_categorical(y_val)
    labels_test = tf.keras.utils.to_categorical(y_test)

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1,
                                                                height_shift_range=0.1,
                                                                horizontal_flip=True,
                                                                vertical_flip=True)

    train_gen = image_gen.flow(x_train, labels_train, batch_size=batch_size, shuffle=True)

    # Create Tf datasets
    # Train data
    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                                   output_types = (tf.float32, tf.float32))
                                                   # output_shapes = ([None, 96, 96, 1], [None, 12]))

    # Validation data
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, labels_val))
    val_dataset = val_dataset.shuffle(buffer_size=len(x_val)).batch(batch_size)
    val_dataset = val_dataset.prefetch(25)

    # Test data
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, labels_test))
    test_dataset = test_dataset.shuffle(buffer_size=len(x_test)).batch(len(x_test)) # feed full test set
    test_dataset = test_dataset.prefetch(25)

    print()
    print("[INFO] Tensorflow data dimensions")
    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    return train_dataset, val_dataset, test_dataset
