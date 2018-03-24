import math


class TickerTable:

    def __init__(self, data):
        self.data = data
        self.train = None
        self.dev = None

    def init_train_dev(self, dev_part=0.2, min_dev_length=0):
        dev_length = math.floor(len(self.data) * dev_part)
        if dev_length < min_dev_length:
            dev_length = min_dev_length

        train_length = len(self.data) - dev_length
        self.train = self.data[0:train_length - 1]
        self.dev = self.data[train_length:]


class TickerRow:

    def __init__(
            self, ticker, date, open_value, high_value, low_value, close_value, volume, ex_dividend, split_ratio,
            adj_open_value, adj_high_value, adj_low_value, adj_close_value, adj_volume
    ):
        self.ticker = ticker
        self.date = date
        self.open = open_value
        self.high = high_value
        self.low = low_value
        self.close = close_value
        self.volume = volume
        self.ex_dividend = ex_dividend
        self.split_ratio = split_ratio
        self.adj_open = adj_open_value
        self.adj_high = adj_high_value
        self.adj_low = adj_low_value
        self.adj_close = adj_close_value
        self.adj_volume = adj_volume
