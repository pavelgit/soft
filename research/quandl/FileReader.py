import csv
from dateutil.parser import parse
from TickerTable import TickerTable, TickerRow


class FileReader:

    def read_csv(self, file_name):
        rows = []
        with open(file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                rows.append(row)

        return rows

    def read_ticker(self, ticker):
        raw_data = self.read_csv('data/tickers/' + ticker + '.csv')
        rows = list(map(
            lambda row: TickerRow(
                ticker=row[0],
                date=parse(row[1]),
                open_value=float(row[2]),
                high_value=float(row[3]),
                low_value=float(row[4]),
                close_value=float(row[5]),
                volume=float(row[6]),
                ex_dividend=float(row[7]),
                split_ratio=float(row[8]),
                adj_open_value=float(row[9]),
                adj_high_value=float(row[10]),
                adj_low_value=float(row[11]),
                adj_close_value=float(row[12]),
                adj_volume=float(row[13])
            ),
            raw_data
        ))
        ticker_data = TickerTable(rows)

        return ticker_data
