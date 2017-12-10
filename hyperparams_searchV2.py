import numpy as np
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.callbacks import CSVLogger
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split

from k_fold_with_model_poc import get_datagen_augment
from k_fold_with_model_poc import get_img_dim


# The default value of 1e-8 for epsilon might not be a good default in general.
# For example, when training an Inception network on ImageNet a current good choice is 1.0 or 0.1.

def inceptionv3_hyp_search(lr=1e-4, epsilon=1e-8, dropout=0.5):
    img_dimension = get_img_dim()
    input_tensor = Input(shape=img_dimension)
    base_model = InceptionV3(include_top=False,
                             weights='imagenet',
                             input_shape=img_dimension)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(rate=dropout)(x)
    output = Dense(12, activation='sigmoid')(x)
    m = Model(input_tensor, output)
    optimizer = Adam(lr=lr, epsilon=epsilon, decay=0)
    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def resnet50_hyp_search(lr=1e-3, epsilon=1e-8, dropout=0.5):
    img_dimension = get_img_dim()
    input_tensor = Input(shape=img_dimension)
    base_model = ResNet50(include_top=False,
                          weights='imagenet',
                          input_shape=img_dimension,
                          pooling='avg')
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Dropout(dropout)(x)
    output = Dense(12, activation='sigmoid')(x)
    m = Model(input_tensor, output)
    optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m


x_loaded = np.load('seedlings_data/numpy_imgs_resized_train.npy')
y_loaded = np.load('seedlings_data/numpy_imgs_train_onehot.npy')
x_train, x_valid, y_train, y_valid = train_test_split(x_loaded, y_loaded, test_size=0.3, random_state=7779,
                                                      shuffle=True)
# alpha between 1e-3 to 5e-5
# batch size between 16 to 32
# dropout between 0.4 to 0.6

epochs = 15

for i in range(1, 20):
    K.clear_session()

    lr = np.power(10, np.random.uniform(-3.5, -3))
    batch_size = np.random.randint(8, 32)
    dropout = np.random.uniform(0.3, 0.8)
    # lr = 1e-3
    # batch_size = 32
    # dropout = 0.5
    # model = inceptionv3_hyp_search(lr=lr, dropout=dropout)
    model = resnet50_hyp_search(lr=lr, dropout=dropout)

    model.summary()

    valid_steps = int(len(x_valid) / batch_size)
    train_steps = int(len(x_train) / batch_size)

    csv_file = ('log_resnet_lr-%f_drop-%f_bsize-%i.log' % (lr, dropout, batch_size))
    callbacks = [CSVLogger(filename=('hyper_tuning/%s' % csv_file))]

    model.fit_generator(get_datagen_augment().flow(x_train, y_train, batch_size), train_steps, epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(x_valid, y_valid))
                        # validation_data=get_datagen_augment().flow(x_valid, y_valid, batch_size),
                        # validation_steps=valid_steps)
