from quandl.FileReader import FileReader
from quandl.nets.BasicNet import BasicNet
from quandl.set_providers.RawSetProvider import RawSetProvider
from quandl.learning_setups import create_tensor_board
import keras


class aapl_2004_basic_net_raw_set_provider_setup:

    def fit(self):
        keras.backend.clear_session()

        file_reader = FileReader()
        data = file_reader.read_ticker('AAPL', '2004-01-01')
        data.init_train_dev()

        tensor_board = create_tensor_board(self.__class__.__name__)
        net = BasicNet(60)

        set_provider = RawSetProvider()
        train_inputs, train_outputs = set_provider.examples_to_sets(data.train, net.day_range)
        dev_inputs, dev_outputs = set_provider.examples_to_sets(data.dev, net.day_range)

        net.fit(train_inputs, train_outputs, dev_inputs, dev_outputs, tensor_board, 100)
