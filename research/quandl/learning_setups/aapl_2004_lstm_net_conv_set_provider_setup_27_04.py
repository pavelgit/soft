from quandl.FileReader import FileReader
from quandl.nets.LSTMNet_27_04_conv import LSTMNet_27_04_conv
from quandl.set_providers.RichInfoSetProvider import RichInfoSetProvider
from quandl.learning_setups import create_tensor_board
import keras


class aapl_2004_lstm_net_conv_set_provider_setup_27_04:

    def fit(self):
        keras.backend.clear_session()

        file_reader = FileReader()
        data = file_reader.read_ticker('AAPL', '2016-01-01')
        #data = file_reader.read_ticker('ATVI', '2017-01-01')
        #data = file_reader.read_ticker('_FAKE')
        day_range = 30
        set_provider = RichInfoSetProvider()

        data.init_train_dev(dev_part=0.2, min_dev_length=day_range+2)

        distribution = set_provider.get_distribution(data.data, day_range)

        print('distribution=', distribution)

        train_inputs, train_outputs = set_provider.examples_to_sets_arrays(data.train, day_range)
        dev_inputs, dev_outputs = set_provider.examples_to_sets_arrays(data.dev, day_range)

        tensor_board = create_tensor_board(self.__class__.__name__)
        net = LSTMNet_27_04_conv(day_range)

        net.fit(train_inputs, train_outputs, dev_inputs, dev_outputs, tensor_board, 100000)
