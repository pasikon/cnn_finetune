import numpy as np
from sklearn.model_selection import KFold
from sklearn import metrics
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


img_height = 300
img_width = 300
img_channels = 3
img_dim = (img_height, img_width, img_channels)


def inceptionv3(img_dim=img_dim):
    input_tensor = Input(shape=img_dim)
    base_model = InceptionV3(include_top=False,
                             weights='imagenet',
                             input_shape=img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation='sigmoid')(x)
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


model = inceptionv3()
model.summary()


batch_size = 32
epochs = 35
n_fold = 5
img_size = (img_height, img_width)
kf = KFold(n_splits=n_fold, shuffle=True)


def train_model_k_fold(model, batch_size, epochs, x, y, test_size, n_fold, kf):
    roc_auc = metrics.roc_auc_score
    # preds_train = np.zeros(len(x), dtype=np.float)
    preds_test = np.zeros(test_size, dtype=np.float)
    train_scores = []
    valid_scores = []

    i = 1

    for train_index, test_index in kf.split(x):   # x to pandas DataFrame
        x_train = x.iloc[train_index]  # train_index to chyba zakres
        x_valid = x.iloc[test_index]  # test_index to chyba zakres
        y_train = y[train_index]
        y_valid = y[test_index]

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
                     ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=1,
                                       verbose=1, min_lr=1e-7),
                     ModelCheckpoint(filepath='inception.fold_' + str(i) + '.hdf5', verbose=1,
                                     save_best_only=True, save_weights_only=True, mode='auto')]

        train_steps = len(x_train) / batch_size
        valid_steps = len(x_valid) / batch_size
        test_steps = test_size / batch_size

        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
                      metrics=['accuracy'])

        model.fit_generator(get_datagen_augment().flow(x_train, y_train, batch_size), train_steps, epochs=epochs, verbose=1,
                            callbacks=callbacks, validation_data=get_datagen_augment().flow(x_valid, y_valid, batch_size),
                            validation_steps=valid_steps)

        model.load_weights(filepath='inception.fold_' + str(i) + '.hdf5')

        print('Running validation predictions on fold {}'.format(i))
        preds_valid = model.predict_generator(generator=get_datagen_augment().flow(x_valid, y_valid, batch_size),
                                              steps=valid_steps, verbose=1)[:, 0]

        print('Running train predictions on fold {}'.format(i))
        preds_train = model.predict_generator(generator=get_datagen_augment().flow(x_train, y_train, batch_size),
                                              steps=train_steps, verbose=1)[:, 0]

        valid_score = roc_auc(y_valid, preds_valid)
        train_score = roc_auc(y_train, preds_train)
        print('Val Score:{} for fold {}'.format(valid_score, i))
        print('Train Score: {} for fold {}'.format(train_score, i))

        valid_scores.append(valid_score)
        train_scores.append(train_score)
        print('Avg Train Score:{0:0.5f}, Val Score:{1:0.5f} after {2:0.5f} folds'.format
              (np.mean(train_scores), np.mean(valid_scores), i))

        print('Running test predictions with fold {}'.format(i))

        preds_test_fold = model.predict_generator(
            generator=get_datagen_noop().flow_from_directory(directory='todoooo', target_size=(300, 300),
                                                             batch_size=batch_size),
            steps=test_steps, verbose=1
        )[:, -1]

        preds_test += preds_test_fold

        print('\n\n')

        i += 1

        if i <= n_fold:
            print('Now beginning training for fold {}\n\n'.format(i))
        else:
            print('Finished training!')

    preds_test /= n_fold

    return preds_test
