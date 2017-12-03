import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

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


# 1. use half of training set
x_loaded = np.load('seedlings_data/numpy_imgs_resized_train.npy')
y_loaded = np.load('seedlings_data/numpy_imgs_train_onehot.npy')


# X_train, X_test, y_train, y_test = train_test_split(x_loaded, y_loaded, test_size=0.3, random_state=456, shuffle=True)


model = KerasClassifier(build_fn=inceptionv3_hyp_search, epochs=2, batch_size=32, verbose=1)
# model.summary

# define the grid search parameters
learn_rate = [1e-3, 1e-4, 7e-5]
epsilon = [1e-8, 0.1, 1]
dropout = [0.4, 0.5, 0.6]
hyperparameters = dict(lr=learn_rate, epsilon=epsilon, dropout=dropout)
grid = GridSearchCV(estimator=model, param_grid=hyperparameters, n_jobs=1, verbose=1)
grid_result = grid.fit(x_loaded, y_loaded)
# summarize results

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
