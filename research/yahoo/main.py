#from yahoo_historical import Fetcher
from fetch import Fetcher
data = Fetcher("AAPL", [2018, 3, 1], [2018, 3, 1], interval='1m')
data.getHistorical().to_csv('aapl.csv', sep=';')