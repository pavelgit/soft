from keras.layers import Input, Dense, BatchNormalization, LSTM, Dropout, Conv1D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import optimizers
import tensorflow as tf


class ClassAccuracyMetric:

    def __init__(self, class_num):
        self.class_num = class_num

    def get_metric_value(self, y_true, y_pred):
        items_of_class = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), self.class_num), tf.float32)
        equal_items = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)), tf.float32)
        items_of_class_sum = tf.reduce_sum(items_of_class)
        successful_of_class_sum = tf.reduce_sum(items_of_class * equal_items)
        return successful_of_class_sum / items_of_class_sum


class LSTMNet_27_04_conv:

    def __init__(self, day_range):
        self.day_range = day_range
        self.model = None
        self.init_model()

    def init_model(self):
        x_input = Input((self.day_range, 6))

        x = x_input

        x_conv_input = BatchNormalization()(x)
        x_conv_output_3 = Conv1D(30, 3, padding='same')(x_conv_input)
        x_conv_output_5 = Conv1D(20, 5, padding='same')(x_conv_input)
        x_conv_output_7 = Conv1D(10, 7, padding='same')(x_conv_input)
        x_conv_output_11 = Conv1D(5, 11, padding='same')(x_conv_input)
        x = Concatenate()([x_conv_input, x_conv_output_3, x_conv_output_5, x_conv_output_7, x_conv_output_11])
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x_conv_input = BatchNormalization()(x)
        x_conv_output_3 = Conv1D(30, 3, padding='same')(x_conv_input)
        x_conv_output_5 = Conv1D(20, 5, padding='same')(x_conv_input)
        x_conv_output_7 = Conv1D(10, 7, padding='same')(x_conv_input)
        x_conv_output_11 = Conv1D(5, 11, padding='same')(x_conv_input)
        x = Concatenate()([x_conv_input, x_conv_output_3, x_conv_output_5, x_conv_output_7, x_conv_output_11])
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x_conv_input = BatchNormalization()(x)
        x_conv_output_3 = Conv1D(30, 3, padding='same')(x_conv_input)
        x_conv_output_5 = Conv1D(20, 5, padding='same')(x_conv_input)
        x_conv_output_7 = Conv1D(10, 7, padding='same')(x_conv_input)
        x_conv_output_11 = Conv1D(5, 11, padding='same')(x_conv_input)
        x = Concatenate()([x_conv_input, x_conv_output_3, x_conv_output_5, x_conv_output_7, x_conv_output_11])
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x = LSTM(150)(x)
        x = Dropout(0.2)(x)

        x = BatchNormalization()(x)
        x = Dense(80)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x = BatchNormalization()(x)
        x = Dense(40)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x = BatchNormalization()(x)
        x = Dense(20)(x)
        x = LeakyReLU()(x)
        x = Dropout(0.2)(x)

        x = Dense(3, activation='softmax')(x)

        self.model = Model(inputs=x_input, outputs=x)

        optimizer = optimizers.Adam(lr=0.01)

        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=[
                'categorical_accuracy',
                ClassAccuracyMetric(0).get_metric_value,
                ClassAccuracyMetric(1).get_metric_value,
                ClassAccuracyMetric(2).get_metric_value
            ]
        )

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

