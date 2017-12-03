import pathlib
import numpy as np
import cv2
from subprocess import check_output
from sklearn.model_selection import KFold
from sklearn import metrics
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import log_loss


def inceptionv3(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionV3(include_top=False,
                             weights='imagenet',
                             input_shape=img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(12, activation='sigmoid')(x)
    m = Model(input_tensor, output)
    return m


def resnet50(img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = ResNet50(include_top=False,
                          weights='imagenet',
                          input_shape=img_dim,
                          pooling='avg')
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Dropout(0.5)(x)
    output = Dense(12, activation='sigmoid')(x)
    m = Model(input_tensor, output)
    return m


def get_datagen_augment():
    generator = ImageDataGenerator(
        rotation_range=7,
        width_shift_range=0.03,
        height_shift_range=0.03,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    return generator


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


def train_model(model, batch_size, epochs, x, y, x_valid, y_valid, name):
    roc_auc = metrics.roc_auc_score
    data_save_path = 'log_no_k_fold/' + name

    valid_steps = int(len(x_valid) / batch_size)
    train_steps = int(len(x) / batch_size)

    callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=1e-4),
                 CSVLogger(data_save_path + '/training_log_no_kfold.log'),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.12, patience=1, cooldown=1,
                                   verbose=1, min_lr=5e-8),
                 ModelCheckpoint(filepath=data_save_path + '/seedl_wgh_ep{epoch:02d}-{val_acc:.5f}.hdf5', verbose=1,
                                 save_best_only=True, save_weights_only=True, mode='auto')]

    model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(get_datagen_augment().flow(x, y, batch_size), train_steps, epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=get_datagen_augment().flow(x_valid, y_valid, batch_size),
                        validation_steps=valid_steps)

    # Make predictions
    predictions_valid = model.predict(x_valid, batch_size=batch_size, verbose=1)

    # Cross-entropy loss score
    score = log_loss(y_valid, predictions_valid)
    print('score pred method 1:' + str(score))

    # ------------------
    print('Running validation predictions\n')
    preds_valid = model.predict_generator(generator=get_datagen_augment().flow(x_valid, y_valid, batch_size,
                                                                               shuffle=False),
                                          steps=valid_steps, verbose=1)

    print('Running train predictions\n')
    preds_train = model.predict_generator(generator=get_datagen_augment().flow(x, y, batch_size,
                                                                               shuffle=False),
                                          steps=train_steps, verbose=1)

    valid_score = roc_auc(y_valid, preds_valid)
    train_score = roc_auc(y, preds_train)
    print('Val Score: {}'.format(valid_score))
    print('Train Score: {}'.format(train_score))


def train_model_k_fold(model, batch_size, epochs, x, y, test_set, n_fold, kf, name):
    data_save_path = 'csvlog/' + name
    validation_batch = 24

    roc_auc = metrics.roc_auc_score

    # preds_train = np.zeros(len(x), dtype=np.float)
    preds_test = np.zeros((test_set.shape[0], y.shape[1]), dtype=np.float)
    train_scores = []
    valid_scores = []

    i = 1

    for train_index, test_index in kf.split(x):
        x_train = x[train_index]
        x_valid = x[test_index]
        y_train = y[train_index]
        y_valid = y[test_index]

        callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, min_delta=1e-4),
                     CSVLogger(data_save_path + '/training_log_fold_' + str(i) + '.log'),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.12, patience=1, cooldown=1,
                                       verbose=1, min_lr=1e-7),
                     ModelCheckpoint(filepath=data_save_path + '/weights_fold_' + str(i) + '.hdf5', verbose=1,
                                     save_best_only=True, save_weights_only=True, mode='auto')]

        train_steps = int(len(x_train) / batch_size)
        valid_steps = int(len(x_valid) / validation_batch)
        test_steps = int(test_set.shape[0] / batch_size)

        model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit_generator(get_datagen_augment().flow(x_train, y_train, batch_size), train_steps, epochs=epochs,
                            verbose=1,
                            callbacks=callbacks,
                            validation_data=get_datagen_augment().flow(x_valid, y_valid, validation_batch),
                            validation_steps=valid_steps)

        model.load_weights(filepath=data_save_path + '/weights_fold_' + str(i) + '.hdf5')

        print('Running validation predictions on fold {}'.format(i))
        preds_valid = model.predict_generator(generator=get_datagen_augment().flow(x_valid, y_valid, validation_batch,
                                                                                   shuffle=False),
                                              steps=valid_steps, verbose=1)

        print('Running train predictions on fold {}'.format(i))
        preds_train = model.predict_generator(generator=get_datagen_augment().flow(x_train, y_train, batch_size,
                                                                                   shuffle=False),
                                              steps=train_steps, verbose=1)

        valid_score = roc_auc(y_valid, preds_valid)
        train_score = roc_auc(y_train, preds_train)
        print('Val Score:{} for fold {}'.format(valid_score, i))
        print('Train Score: {} for fold {}'.format(train_score, i))

        valid_scores.append(valid_score)
        train_scores.append(train_score)
        print('----------------\n')
        print('Avg Train Score:{0:0.5f}, Val Score:{1:0.5f} after {2:0.5f} folds'.format
              (np.mean(train_scores), np.mean(valid_scores), i))

        print('Running test predictions with fold {}'.format(i))

        preds_test_fold = model.predict(test_set, batch_size=batch_size, verbose=1)

        preds_test += preds_test_fold

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
    kf = KFold(n_splits=n_fold, shuffle=True)
    x_loaded = np.load('seedlings_data/numpy_imgs_resized_train.npy')
    y_loaded = np.load('seedlings_data/numpy_imgs_train_onehot.npy')

    img_dimension = get_img_dim()

    # model = inceptionv3(img_dimension)
    model = resnet50(img_dimension)
    model.summary()

    batch_size = 24
    epochs = 25

    run_name = 'ResNet50_preds_kfold10ep25_cat_cross_ent'
    pathlib.Path('csvlog/' + run_name).mkdir(parents=True, exist_ok=True)

    test_set = ld_test_set()

    model_k_fold = train_model_k_fold(model=model, batch_size=batch_size, epochs=epochs, x=x_loaded, y=y_loaded,
                                      test_set=test_set, n_fold=n_fold, kf=kf, name=run_name)
    np.save('seedl_subms/' + run_name + '.npy', model_k_fold)

    print(model_k_fold)
