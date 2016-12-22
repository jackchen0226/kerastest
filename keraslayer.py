from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class MyLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

e):
        input_dim = input_shape[1]
        initial_weight_value = np.ones((input_dim, output_dim))
        self.W = K.variable(initial_weight_value, name="test")
        self.trainable_weights = [self.W]
        super(MyLayer, self).build()

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

