from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import backend as K

class InitialNet:

    def get_model(self, input_shape):
        x_input = Input(input_shape)

        x = x_input



        model = Model(inputs=x_input, outputs=x)

        return model