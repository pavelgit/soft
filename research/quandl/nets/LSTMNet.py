from keras.layers import Input, Dense, BatchNormalization, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import optimizers

class LSTMNet:

    def __init__(self, day_range):
        self.day_range = day_range
        self.model = None
        self.init_model()

    def init_model(self):
        x_input = Input((self.day_range, 1))

        x = x_input

        x = LSTM(100)(x)

        x = BatchNormalization()(x)
        x = Dense(80)(x)
        x = LeakyReLU()(x)

        x = BatchNormalization()(x)
        x = Dense(40)(x)
        x = LeakyReLU()(x)

        x = BatchNormalization()(x)
        x = Dense(20)(x)
        x = LeakyReLU()(x)

        x = Dense(3, activation='softmax')(x)

        self.model = Model(inputs=x_input, outputs=x)

        optimizer = optimizers.Adam(lr=0.01)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=[
            'accuracy'
        ])

    def fit(self, train_inputs, train_outputs, dev_inputs, dev_outputs, tensor_board, epochs=100):
        self.model.fit(
            train_inputs,
            train_outputs,
            validation_data=(dev_inputs, dev_outputs),
            epochs=epochs,
            callbacks=[tensor_board]
        )

    def evaluate(self, inputs, outputs):
        return self.model.evaluate(inputs, outputs)

