import tensorflow as tf

def AlexNet(input_shape, num_classes):
    """
    """
    inputs = tf.keras.layers.Input(shape=input_shape, name="input_images")

    # layer 1
    conv_1 = tf.keras.layers.Conv2D(filters=96,
                                    kernel_size=(11, 11),
                                    strides=4,
                                    padding="valid",
                                    activation=tf.keras.activations.relu)(inputs)

    max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                           strides=2,
                                           padding="valid")(conv_1)

    batchnorm_1 = tf.keras.layers.BatchNormalization()(max_pool_1)

    # layer 2
    conv_2 = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=(5, 5),
                                    strides=1,
                                    padding="same",
                                    activation=tf.keras.activations.relu)(batchnorm_1)

    max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                           strides=2,
                                           padding="same")(conv_2)

    batchnorm_2 = tf.keras.layers.BatchNormalization()(max_pool_2)

    # layer 3
    conv_3 = tf.keras.layers.Conv2D(filters=384,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           activation=tf.keras.activations.relu)(batchnorm_2)

    # layer 4
    conv_4 = tf.keras.layers.Conv2D(filters=384,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           activation=tf.keras.activations.relu)(conv_3)

    # layer 5
    conv_5 = tf.keras.layers.Conv2D(filters=256,
                           kernel_size=(3, 3),
                           strides=1,
                           padding="same",
                           activation=tf.keras.activations.relu)(conv_4)

    max_pool_5 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                           strides=2,
                                           padding="same")(conv_5)

    batchnorm_5 = tf.keras.layers.BatchNormalization()(max_pool_5)

    # layer 6
    flatten_6 = tf.keras.layers.Flatten()(batchnorm_5)

    dense_6 = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu)(flatten_6)

    dropout_6 = tf.keras.layers.Dropout(rate=0.2)(dense_6)

    # layer 7
    dense_7 = tf.keras.layers.Dense(units=4096, activation=tf.keras.activations.relu)(dropout_6)

    dropout_7 = tf.keras.layers.Dropout(rate=0.2)(dense_7)

    # layer 8
    # output is (128, 4) --> (batch_size, num_classes)
    dense_8 = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)(dropout_7)

    return tf.keras.Model(inputs=inputs, outputs=dense_8)
