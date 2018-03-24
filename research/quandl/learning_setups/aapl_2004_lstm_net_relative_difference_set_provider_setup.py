from quandl.FileReader import FileReader
from quandl.nets.LSTMNet import LSTMNet
from quandl.set_providers.RelativeDifferenceSetProvider import RelativeDifferenceSetProvider
from quandl.learning_setups import create_tensor_board
import keras


class aapl_2004_lstm_net_relative_difference_set_provider_setup:

    def fit(self):
        keras.backend.clear_session()

        file_reader = FileReader()
        data = file_reader.read_ticker('AAPL', '2017-11-01')
        #data = file_reader.read_ticker('_FAKE')
        day_range = 30
        set_provider = RelativeDifferenceSetProvider()

        data.init_train_dev(dev_part=0.0, min_dev_length=day_range+2)

        raw_inputs, raw_outputs = set_provider.examples_to_sets_raw(data.data, day_range)

        distribution = set_provider.get_distribution(data.data, day_range)

        print('distribution=', distribution)

        train_inputs, train_outputs = set_provider.examples_to_sets_arrays(data.train, day_range)
        dev_inputs, dev_outputs = set_provider.examples_to_sets_arrays(data.dev, day_range)

        tensor_board = create_tensor_board(self.__class__.__name__)
        net = LSTMNet(30)

        net.fit(train_inputs, train_outputs, dev_inputs, dev_outputs, tensor_board, 100)
