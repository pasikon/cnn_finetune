import numpy as np
import pathlib
from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from sklearn.model_selection import train_test_split
from k_fold_with_model_poc import get_datagen_augment, get_img_dim, ld_test_set
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, TensorBoard, Callback, LambdaCallback
from data_load_memory import load_train_data_dir, load_problematic_pics
from time import gmtime, strftime


class ProblematicPicsCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting PROBLEMATIC PICS loss: {}, acc: {}\n'.format(loss, acc))


problematic_pics, problematic_pics_true = load_problematic_pics()
problematic_pics = problematic_pics / 255.

x_loaded = np.load('seedlings_data/numpy_imgs_resized_train.npy')
x_loaded = np.array(x_loaded, np.float32) / 255.
y_loaded = np.load('seedlings_data/numpy_imgs_train_onehot.npy')

x_loaded_segm, y_loaded_segm = load_train_data_dir('seedlings_data', 'train_segmented')

x_train, x_test, y_train, y_test = train_test_split(x_loaded, y_loaded, test_size=0.3, random_state=456, shuffle=True)

batch_size = 32
train_steps = len(x_train) / batch_size
valid_steps = len(x_test) / batch_size
run_date_time_str = strftime("run_%Y-%m-%d-%H_%M_%S", gmtime())
pathlib.Path('FINETUNING/' + run_date_time_str).mkdir(parents=True, exist_ok=True)


# create the base pre-trained model
def inceptionv3_model():
    input_tensor = Input(shape=get_img_dim())  # this assumes K.image_data_format() == 'channels_last'

    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=get_img_dim())

    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    # and a logistic layer
    predictions = Dense(12, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=input_tensor, outputs=predictions)
    return base_model, model


base_model, model = inceptionv3_model()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

# train the model on the new data for a few epochs
model.fit_generator(generator=get_datagen_augment().flow(x=x_train, y=y_train, batch_size=batch_size),
                    steps_per_epoch=train_steps, epochs=6, validation_data=(x_test, y_test), workers=12,
                    max_queue_size=100)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in base_model.layers[:249]:
    # layer.trainable = False
    layer.trainable = True
for layer in base_model.layers[249:]:
    layer.trainable = True

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.summary()

callbacks = [
    ModelCheckpoint(filepath='FINETUNING/' + run_date_time_str + '/weights.{epoch:02d}-loss_{val_loss:.5f}.hdf5',
                    verbose=1, save_best_only=True, save_weights_only=True, mode='auto'),
    ProblematicPicsCallback((problematic_pics, problematic_pics_true)),
    TensorBoard(log_dir='FINETUNING/' + run_date_time_str),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, cooldown=1, verbose=1),
    CSVLogger('FINETUNING/' + run_date_time_str + '/training_log.csv')
]

# train model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(generator=get_datagen_augment().flow(x=x_train, y=y_train, batch_size=batch_size),
                    steps_per_epoch=train_steps, epochs=80, callbacks=callbacks,
                    validation_data=(x_test, y_test), workers=12,
                    max_queue_size=100)

score_test = model.evaluate(x=x_test, y=y_test, verbose=0)
score_train = model.evaluate(x=x_train, y=y_train, verbose=0)
score_problematic_pics = model.evaluate(x=problematic_pics, y=problematic_pics_true, verbose=0)
score_train_aug = model.evaluate_generator(generator=get_datagen_augment().flow(x_train, y_train, batch_size,
                                                                                shuffle=False),
                                           steps=train_steps, max_queue_size=100, workers=12)
score_test_aug = model.evaluate_generator(generator=get_datagen_augment().flow(x_test, y_test, batch_size,
                                                                               shuffle=False),
                                          steps=valid_steps, max_queue_size=100, workers=12)

print('Train Score: {}'.format(score_train))
print('Test Score: {}'.format(score_test))
print('Train Score AUG: {}'.format(score_train_aug))
print('Test Score AUG: {}'.format(score_test_aug))
print('Problematic pics score: {}'.format(score_problematic_pics))

print('Loading competition set...')
c_set = ld_test_set()
c_set = np.array(c_set, np.float32) / 255.
print('Predicting competition set...')
c_set_preds = model.predict(c_set, batch_size=batch_size, verbose=0)
print('Saving competition set...')
np.save('seedl_subms/FINETUNE.npy', c_set_preds)
print('Competition set sample:')
print(str(c_set_preds[332]))
