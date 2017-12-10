import pathlib
from subprocess import check_output

import cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, RandomizedSearchCV


def xception(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = Xception(include_top=False, weights='imagenet',
                          input_shape=img_dim,
                          pooling='avg')
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    # x = GlobalAveragePooling2D()(x)
    x = Dropout(0.75)(x)
    output = Dense(12, activation='sigmoid')(x)
    m = Model(input_tensor, output)
    m.compile(optimizer=Adam(lr=7.5686080109320267e-05), loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def inceptionv3(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionV3(include_top=False,
                             weights='imagenet',
                             input_shape=img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.56858104896609896)(x)
    output = Dense(12, activation='sigmoid')(x)
    m = Model(input_tensor, output)
    m.compile(optimizer=Adam(lr=8.9141954823086827e-05), loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def resnet50(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = ResNet50(include_top=False,
                          weights='imagenet',
                          input_shape=img_dim,
                          pooling='avg')
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    # x = Dropout(0.47)(x)
    output = Dense(12, activation='sigmoid')(x)
    m = Model(input_tensor, output)
    optimizer = SGD(decay=1e-6, momentum=0.9, nesterov=True, lr=8e-4)
    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def get_datagen_augment():
    generator = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.13,
        height_shift_range=0.13,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    return generator


def random_crop(image):
    crop_rate = np.random.randint(0, 25)
    crop_axis_x = np.random.randint(2)
    crop_axis_y = np.random.randint(2)
    if crop_axis_x == 0 and crop_axis_y == 0:
        img = image[0:(image.shape[0] - crop_rate), 0:(image.shape[1] - crop_rate)]
    elif crop_axis_x == 0 and crop_axis_y == 1:
        img = image[0:(image.shape[0] - crop_rate), crop_rate:image.shape[1]]
    elif crop_axis_x == 1 and crop_axis_y == 0:
        img = image[crop_rate:image.shape[0], 0:(image.shape[1] - crop_rate)]
    else:
        img = image[crop_rate:image.shape[0], crop_rate:image.shape[1]]
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    return img


def ld_test_set():
    fpa = []
    fpa.append(check_output(["ls", "seedlings_data/test/"]).decode("utf8").strip().split("\n"))

    fpa = np.array(fpa)
    fpa = np.core.defchararray.add("seedlings_data/test/", fpa)
    fpa = fpa.squeeze()

    imgs = []

    for im_path in fpa:
        if 'Thumbs.db' not in im_path:
            # print(im_path)
            imageCV = cv2.imread(im_path)
            imgs.append(cv2.resize(imageCV, (300, 300), interpolation=cv2.INTER_AREA))

    imgs = np.array(imgs)

    return imgs


def get_img_dim():
    img_height = 300
    img_width = 300
    img_channels = 3
    img_dim = (img_height, img_width, img_channels)
    return img_dim


if __name__ == '__main__':
    x_loaded = np.load('seedlings_data/numpy_imgs_resized_train.npy')
    y_loaded = np.load('seedlings_data/numpy_imgs_train_onehot.npy')

    x_train, x_test, y_train, y_test = train_test_split(x_loaded, y_loaded, test_size=0.2, random_state=456, shuffle=True)

    img_dimension = get_img_dim()

    model = inceptionv3(img_dimension)

    model.summary()

    batch_size = 32
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

    train_steps = x_train.shape[0] / batch_size
    model.fit_generator(get_datagen_augment().flow(x_train, y_train, batch_size), validation_steps=train_steps,
                        epochs=50,
                        verbose=1,
                        callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
                        validation_data=(x_test, y_test))
