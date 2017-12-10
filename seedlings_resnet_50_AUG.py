# -*- coding: utf-8 -*-

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Flatten, \
    merge, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras import backend as K

from sklearn.metrics import log_loss


def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1', trainable=trainable)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x


def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None, start_fresh=False):
    """
    Resnet 50 Model for Keras

    Model Schema is based on
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

    ImageNet Pretrained Weights
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', trainable=False)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', trainable=False)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', trainable=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', trainable=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', trainable=False)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', trainable=False)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', trainable=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', trainable=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', trainable=False)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x_fc)
    print('Dimension ordering (tf for Tensorflow): ' + str(K.image_dim_ordering()))
    # For pre-trained ImageNet load weights here
    if start_fresh:
        weights_path = 'imagenet_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
        print('Loading FRESH ImageNet weights from file: ' + weights_path)
        model = model.load_weights(weights_path)
        print('Loading FRESH weights finished...')

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)

    # For seedlings loading model here
    if not start_fresh:
        weights_path = 'seedl_chkp/seedl_wgh_AUG_lr9e-4_rot7-ep28-0.97068.hdf5'
        print('Loading NOT FRESH weights from file: ' + weights_path)
        model.load_weights(weights_path)
        print('Loading NOT FRESH weights finished...')

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


if __name__ == '__main__':
    img_rows, img_cols = 300, 300  # Resolution of inputs
    channel = 3
    num_classes = 12
    batch_size = 32
    epochs = 64

    x_loaded = np.load('seedlings_data/numpy_imgs_resized_train.npy')
    y_loaded = np.load('seedlings_data/numpy_imgs_train_onehot.npy')

    x_loaded, y_loaded = unison_shuffled_copies(x_loaded, y_loaded)

    X_train = x_loaded[0:4100, :]
    Y_train = y_loaded[0:4100, :]
    X_valid = x_loaded[4101:4750, :]
    Y_valid = y_loaded[4101:4750, :]

    datagen_aug = ImageDataGenerator(
        # zoom_range=0.1,
        rotation_range=7,
        fill_mode='constant',
        # width_shift_range=0.2,
        vertical_flip=True,
        # height_shift_range=0.2,
        horizontal_flip=True
    )

    # checkpoint
    chk_fp = "seedl_chkp/seedl_wgh_AUG_NT1234_redlr_lr1e-3_rot7-ep{epoch:02d}-{val_acc:.5f}.hdf5"
    checkpoint = ModelCheckpoint(chk_fp, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    csv_logger = CSVLogger('csvlog/training.log')
    # reduce lr
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    callbacks_list = [checkpoint, reduce_lr, csv_logger]

    # Load our model
    model = resnet50_model(img_rows, img_cols, channel, num_classes, start_fresh=True)

    # Start Fine-tuning

    steps_per_epoch = int(X_train.shape[0] / batch_size)
    print('Steps per epoch: %s' % steps_per_epoch)

    model.fit_generator(
        # X_train, Y_train,
        datagen_aug.flow(
            X_train, Y_train,
            batch_size=batch_size,
            shuffle=True
            # save_to_dir='augmented_pics'
        ),
        epochs=epochs,
        workers=12,
        verbose=1,
        steps_per_epoch=steps_per_epoch,  # Steps per epoch = samples_training_set/batch_size
        validation_data=(X_valid, Y_valid),
        callbacks=callbacks_list
    )

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    print(score)
