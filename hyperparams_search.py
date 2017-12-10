import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from keras import backend as K

from k_fold_with_model_poc import get_img_dim
import scipy.stats as st
from scipy.stats.distributions import expon, uniform
from sklearn.model_selection import ParameterSampler


# The default value of 1e-8 for epsilon might not be a good default in general.
# For example, when training an Inception network on ImageNet a current good choice is 1.0 or 0.1.

def inceptionv3_hyp_search(lr=1e-4, dropout=0.5):
    K.clear_session()
    print('lr: ' + str(lr))
    print('dropout: ' + str(dropout))
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
    optimizer = Adam(lr=lr, decay=0)
    m.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return m


# 1. use half of training set
x_loaded = np.load('seedlings_data/numpy_imgs_resized_train.npy')
y_loaded = np.load('seedlings_data/numpy_imgs_train_onehot.npy')

X_train, X_test, y_train, y_test = train_test_split(x_loaded, y_loaded, test_size=0.2, random_state=456, shuffle=True)

model = KerasClassifier(build_fn=inceptionv3_hyp_search, epochs=15, batch_size=32, verbose=1)

hyperparameters_dist = dict(lr=st.uniform(3e-5, 1e-4),
                            dropout=st.uniform(3e-1, 9e-1))

random = RandomizedSearchCV(estimator=model,
                            param_distributions=hyperparameters_dist,
                            n_jobs=1,
                            verbose=1,
                            cv=3,
                            n_iter=15)
# test na 1/5 danych
rnd_result = random.fit(X_test, y_test)

print("Best: %f using %s" % (rnd_result.best_score_, rnd_result.best_params_))
means = rnd_result.cv_results_['mean_test_score']
stds = rnd_result.cv_results_['std_test_score']
params = rnd_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
