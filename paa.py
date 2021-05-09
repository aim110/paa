# coding: utf-8
import yfinance  # kinda docs: https://pypi.org/project/yfinance/
import pandas as pd  # https://pandas.pydata.org/docs/index.html
pd.options.mode.chained_assignment = None  # default='warn' to make my excessive data copying sleep well

from datetime import date


def MOM(last_price, sma):
    return last_price/sma - 1


def SMA(ticker_cleaned, L=12):
    """ wma: https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average """
    weighted = pd.DataFrame({'Price': ticker_cleaned.dropna()["Close"], 'Weight': range(L)}, index=ticker_cleaned.index)
    return sum(weighted["Weight"] * weighted["Price"]) / sum(weighted["Weight"])


def get_month(row):
    return str(row["Day"])[:7]


def fetch_all_ticker_data(tickers):
    today = date.today()
    start = date(year = today.year - 1, month = today.month - 1, day = today.day)
    return yfinance.download(tickers, start=str(start), end=str(today), interval='1mo', group_by='ticker')


def cleanup_ticker_data(d, lookback):
    d = d.dropna()
    d["Day"] = d.index
    d["Mo"] = d.apply(get_month, axis=1)
    # leave 1 record per month and take a last one

    d = d.sort_values("Day").drop_duplicates(['Mo'], keep='last')
    d = d.drop(["Mo", "High", "Low", "Adj Close", "Volume"], axis=1)
    return d.tail(lookback)


def BF(N, n, a):
    n1 = a*N/4.
    return (N - n) / (N - n1)


def print_sorted_mom_tickers(tm_list):
    for t, mom in tm_list:
        print("{}: {}".format(t, mom))


def print_sorted_mom_filtered(tickers, all_momentum_sorted):
    for t, m in [(t,m) for t, m in all_momentum_sorted if t in set(tickers)]:
        print("{}: {}".format(t, m))


def main():
    risky = set(['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'HYG', 'LQD', 'TLT']) # NB: SCZ looks quite interesting!!!
    canary = set(['AGG', 'EEM'])
    safe = set(['IEF', 'SHY', 'LQD', 'TIPS', 'VGSH'])

    all_tickers = {*risky, *canary, *safe}
    df = fetch_all_ticker_data(all_tickers)

    # PAA1 strategy:
    L = 12  # Lookback length (month)
    A = 1   # Protection range (0: low, 1: medium, 2: high)
    N = len(risky)  # Universum
    TOP = 6  # Number of assets in rotation
    # Paper strategy comparison: page 9

    ticker_mom = {}
    for t in all_tickers:
        ticker_data = df[t].copy(deep=True)
        cleaned_data = cleanup_ticker_data(ticker_data, L)

        sma = SMA(cleaned_data, L)  # TBD: not sure if I am doing that correctly. Do I need 13 points for SMA12? Should I include p0 to SMA calculation?
        last_price = cleaned_data.tail(1).iloc[0]["Close"]

        momentum = MOM(last_price, sma)
        ticker_mom[t] = momentum

    sorted_mom = sorted(ticker_mom.items(), key=lambda x: x[1])
    print("\nAll tickers momentum:")
    print_sorted_mom_tickers(sorted_mom)

    positive_momentum_assets = 0
    for t in risky:
        if ticker_mom[t] > 0:
            positive_momentum_assets += 1
    print("\nPositive momentum cnt for risky: ", positive_momentum_assets)

    top = [(t,m) for t, m in sorted_mom if m > 0 and t in risky][-TOP:]
    print("\nTop:")
    print_sorted_mom_tickers(top)

    bf = BF(N, positive_momentum_assets, A)
    risky_asset_share = (1.0 - bf) / TOP
    print("\nShare of each risky asset: ", round(risky_asset_share, 2))

    print("\nBond fraction: ", round(bf, 2))

    print("\nSafe assets momentum:")
    print_sorted_mom_filtered(safe, sorted_mom)


if __name__ == '__main__':
    main()
