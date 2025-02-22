import yfinance as yf

class TickerDownloader(object):
    def __init__(self, tickers, from_date, to_date):
        self._tickers = tickers
        self._from_date = from_date
        self._to_date = to_date
        #self._td = None

    def _validate(self, td):
        """ td - ticker data """
        # check columns - contains all tickers
        extracted_tickers = set(map(lambda x: x[0], td.columns.tolist()))
        assert len(extracted_tickers) == len(self._tickers)
        # check dates - contains all dates i wanted, Are the holidays are skipped in data?
        delta = (self._to_date - self._from_date).days
        assert len(td.index) > delta * 5 / 7 - 10  # week days minus 10 holidays for a year
        print("Ticker dataset validation: ok")

    def download(self):
        td = yf.download(self._tickers, start=str(self._from_date), end=str(self._to_date), interval='1d', group_by='ticker')
        self._validate(td)
        return td
