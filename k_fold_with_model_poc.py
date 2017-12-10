import pathlib
from subprocess import check_output

import cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold


def xception(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = Xception(include_top=False, weights='imagenet',
                          input_shape=img_dim,
                          pooling='avg')
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Dropout(0.75)(x)
    output = Dense(12, activation='softmax')(x)
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
    x = Dropout(0.81463030250387169)(x)
    output = Dense(12, activation='softmax')(x)
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
    output = Dense(12, activation='softmax')(x)
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


def validation_rnd_crop_generator():
    generator = ImageDataGenerator(
        preprocessing_function=random_crop
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


def get_datagen_noop():
    generator = ImageDataGenerator()
    return generator


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


def train_model_k_fold(model, batch_size, epochs, x, y, test_set, n_fold, kf, name):
    data_save_path = 'csvlog/' + name

    preds_test = np.zeros((test_set.shape[0], y.shape[1]), dtype=np.float)
    train_acc_scores = []
    valid_acc_scores = []

    i = 1

    for train_index, test_index in kf.split(x):
        x_train = x[train_index]
        x_valid = x[test_index]
        y_train = y[train_index]
        y_valid = y[test_index]

        callbacks = [EarlyStopping(monitor='val_loss', patience=8, verbose=1),
                     CSVLogger(data_save_path + '/training_log_fold_' + str(i) + '.log'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, cooldown=1,
                                       verbose=1),
                     ModelCheckpoint(filepath=data_save_path + '/weights_fold_' + str(i) + '.hdf5', verbose=1,
                                     save_best_only=True, save_weights_only=True, mode='auto')]

        train_steps = len(x_train) / batch_size
        valid_steps = len(x_valid) / batch_size
        test_steps = test_set.shape[0] / batch_size

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])

        model.fit_generator(get_datagen_augment().flow(x_train, y_train, batch_size), steps_per_epoch=train_steps,
                            epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=get_datagen_augment().flow(x_valid, y_valid, batch_size),
                            validation_steps=valid_steps, workers=12, max_queue_size=100)

        model.load_weights(filepath=data_save_path + '/weights_fold_' + str(i) + '.hdf5')

        print('Running validation predictions on fold {}'.format(i))
        score_valid = model.evaluate_generator(generator=get_datagen_augment().flow(x_valid, y_valid, batch_size,
                                                                                    shuffle=False),
                                               steps=valid_steps, workers=12, max_queue_size=100)

        print('Running train predictions on fold {}'.format(i))
        score_train = model.evaluate_generator(generator=get_datagen_augment().flow(x_train, y_train, batch_size,
                                                                                    shuffle=False),
                                               steps=train_steps, workers=12, max_queue_size=100)

        print('validation (aug) loss and cat_accuracy: {} for fold {}'.format(score_valid, i))
        print('train (aug) loss and cat_accuracy: {} for fold {}'.format(score_train, i))

        valid_acc_scores.append(score_valid[1])
        train_acc_scores.append(score_train[1])

        print('Avg Train accuracy:{0:0.5f}, Val accuracy:{1:0.5f} after {2:0.5f} folds'.format
              (np.mean(train_acc_scores), np.mean(valid_acc_scores), i))

        print('Running test set predictions with fold {}'.format(i))

        preds_test_fold = model.predict(test_set, batch_size=batch_size, verbose=0)
        for pi in range(0, 3):
            print('Test set predictions random cropping iteration: {}'.format(pi))
            preds_test_fold += model.predict_generator(
                generator=validation_rnd_crop_generator().flow(x=test_set, batch_size=batch_size, shuffle=False),
                steps=test_steps,
                verbose=0,
                workers=12,
                max_queue_size=100
            )

        preds_test += (preds_test_fold / 4)

        print('\n\n')

        i += 1

        if i <= n_fold:
            print('Now beginning training for fold {}\n\n'.format(i))
        else:
            print('Finished training!')

    preds_test /= n_fold

    return preds_test


def get_img_dim():
    img_height = 300
    img_width = 300
    img_channels = 3
    img_dim = (img_height, img_width, img_channels)
    return img_dim


if __name__ == '__main__':
    n_fold = 10
    batch_size = 32
    epochs = 1

    kf = KFold(n_splits=n_fold, shuffle=True)
    x_loaded = np.load('seedlings_data/numpy_imgs_resized_train.npy')
    y_loaded = np.load('seedlings_data/numpy_imgs_train_onehot.npy')

    x_loaded = np.array(x_loaded, np.float32) / 255.

    test_set = ld_test_set()
    test_set = np.array(test_set, np.float32) / 255.

    img_dimension = get_img_dim()

    model = inceptionv3(img_dimension)
    # model = xception(img_dimension)
    # model = resnet50(img_dimension)
    model.summary()

    run_name = 'del_me'
    pathlib.Path('csvlog/' + run_name).mkdir(parents=True, exist_ok=True)

    model_k_fold = train_model_k_fold(model=model, batch_size=batch_size, epochs=epochs, x=x_loaded, y=y_loaded,
                                      test_set=test_set, n_fold=n_fold, kf=kf, name=run_name)
    np.save('seedl_subms/' + run_name + '.npy', model_k_fold)

    print(model_k_fold)
