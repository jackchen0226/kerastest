'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function
import numpy as np
from scipy.misc.pilutil import imsave
np.random.seed(9180)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers.noise import GaussianNoise, GaussianDropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K


batch_size = 128
nb_classes = 10
nb_epoch = 1

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#print(X_train)
#imsave("startset.png", X_train)
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

from keras.engine.topology import Layer
'''
class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.ones((input_dim, self.output_dim))
        self.W = K.variable(initial_weight_value, name="test")
        self.trainable_weights = [self.W]
        super(MyLayer, self).build(input_shape)

    def call(self, x, input_shape, mask=None):
        return K.any(x, self.W, keepdims=True)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], output_dim)
'''

class NoiseLayer(Layer):
    def __init__(self, sigma=1.0, **kwargs):
        self.supports_masking = True
        self.sigma = sigma
        self.uses_learning_phase = True
        super(NoiseLayer, self).__init__(**kwargs)

    def call(self, x, mask=None):
	# noise is a random normal distribution curve
        noise = K.random_normal(shape=K.shape(x),
                                      mean=0.,
                                      std=self.sigma)
	# noise_x is the noise tensor combined with the input tensor
        noise_x = x + noise        
	# Then, noise_x is converted down to a list (tensor > numpy array > list)
        noise_x = K.eval(noise_x)
        print(noise_x)
        noise_x = noise_x.tolist()
        print(noise_x)
	# Limit final values, a greyscale pixel can't be lighter than 1.0 (white) or darker than 0.0 (black)
        for i in range(len(noise_x)):
            for j in range(len(noise_x[i])):
                if j > 1.0:
                    noise_x[i][j] = 1.0
                elif j < 0.0:
                    noise_x[i][j] = 0.0

        noise_x.asarray(noise_x)
        y = K.placeholder(shape=(512,60000))
        # Make sure noise_x becomes a tensor here
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {'sigma': self.sigma}
        base_config = super(GaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

model = Sequential()
model.add(GaussianNoise(3.0, input_shape=(784,)))
#model.add(Dense(512, input_shape=(784,)))
model.add(Dense(512))
#model.add(NoiseLayer(1.0))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512, name="2nd_dense"))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

from datetime import datetime
output_layer_name = "gaussiannoise_1"
intermediate_layer_model = Model(input=model.input, output=model.get_layer(output_layer_name).output)
intermediate_output = intermediate_layer_model.predict(X_train)
print(intermediate_output)
imsave('output_images/outfile {} {:%B %d,%H:%M:%S}.png'.format(output_layer_name, datetime.now()), intermediate_output)
