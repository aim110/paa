# coding: utf-8
import yfinance as yf  # kinda docs: https://pypi.org/project/yfinance/
import pandas as pd  # https://pandas.pydata.org/docs/index.html
pd.options.mode.chained_assignment = None  # default='warn' to make my excessive data copying sleep well

import argparse
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
    start = date(year = today.year - 1, month = today.month - 1, day = 1)
    return yf.download(tickers, start=str(start), end=str(today), interval='1mo', group_by='ticker')


def cleanup_ticker_data(d, lookback):
    d = d.dropna()
    d["Day"] = d.index
    d["Mo"] = d.apply(get_month, axis=1)
    # leave 1 record per month and take a last one

    d = d.sort_values("Day").drop_duplicates(['Mo'], keep='last')
    d = d.drop(["Mo", "High", "Low", "Adj Close", "Volume"], axis=1)
    return d.tail(lookback)


def BF(N, n, a):
    """
    calc bond fraction
    """
    n1 = a*N/4.
    return (N - n) / (N - n1)


def print_sorted_mom_tickers(tm_list):
    for t, mom in tm_list:
        print("{}: {}".format(t, mom))


def print_sorted_mom_filtered(tickers, all_momentum_sorted):
    for t, m in [(t,m) for t, m in all_momentum_sorted if t in set(tickers)]:
        print("{}: {}".format(t, m))


def get_last_prices(tickers_list):
    data = yf.download(tickers=tickers_list, period='2h', interval='5m')
    tail = data.dropna().tail(1)["Close"]
    return {ticker: tail[ticker][0] for ticker in tail}


def read_ticker_state(fn):
    return pd.read_csv(fn, header=None, names=["ticker", "amount"], index_col="ticker")


def prepare_orders(ticker_current, target_tickers, ticker_to_sum):
    """
    orders to buy and to sell
    """
    all_tickers = set(ticker_current.index) | set(target_tickers) 
    last_price = get_last_prices(all_tickers)
    #print("LP: {}".format(last_price))

    orders = []
    # BUY PART
    for ticker in target_tickers:
        #print("Ticker: {}".format(ticker))
        try:
            current_shares = ticker_current.at[ticker, "amount"]
        except KeyError:
            current_shares = 0
        target_shares = ticker_to_sum[ticker] / last_price[ticker]
        #print("curr: {}\ttarget_shares: {}".format(current_shares, target_shares))

        amount = target_shares - current_shares 
        order_type = 'buy' if amount > 0 else 'sell'
        orders.append((order_type, ticker, amount))

    # Sell shares that not in target but currently available
    to_sell = set(ticker_current.index) - set(target_tickers)
    for t in to_sell:
        orders.append(('sell', t, 'all'))


    return orders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--amount', type=int, help="calculate shares according to that amount of money", required=True)
    parser.add_argument('--current_fn', type=str, help="csv files with current positions", required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    paa_risky = set(['VTI', 'IWM', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'HYG', 'LQD', 'TLT']) # NB: SPY - original, I got VTI instead
    canary = set(['AGG', 'EEM'])
    iamcurious = set('EFV RWO IWN SCZ EFA IVE'.split()) # TBD: move to separate script for wondering
    sectoral = set('VAW VOX XLE VFH VIS VGT VDC VNQ VPU VHT VCR'.split())  # energy communcation energy finance industrials tech staples realEstate utilities healthcare discretionary
    # is VDE better than XLE?
    safe = set(['IEF', 'SHY', 'LQD', 'TIP', 'VGSH', 'VCSH', 'IAGG', "STIP"]) # STIP

    all_tickers = {*paa_risky, *canary, *safe, *iamcurious, *sectoral}
    df = fetch_all_ticker_data(all_tickers)

    # PAA1 strategy:
    L = 12  # Lookback length (month)
    A = 1   # Protection range (0: low, 1: medium, 2: high)
    N = len(paa_risky)  # Universum
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
    for t, mom in sorted_mom:
        extra = ' ' if t not in paa_risky else '*'
        print("{} {}: {}".format(extra, t, mom))

    positive_momentum_assets = 0
    for t in paa_risky:
        if ticker_mom[t] > 0:
            positive_momentum_assets += 1
    print("\nPositive momentum cnt for paa risky: ", positive_momentum_assets)

    top = [(t,m) for t, m in sorted_mom if m > 0 and t in paa_risky][-TOP:]
    print("\nTop risky assets for PAA:")
    print_sorted_mom_tickers(top)

    bf = BF(N, positive_momentum_assets, A)
    risky_asset_share = (1.0 - bf) / TOP
    print("\nShare of each risky asset: ", round(risky_asset_share, 2))
    risky_target = round(args.amount * risky_asset_share, 1)

    print("\nBond fraction: ", round(bf, 2))

    print("\nSafe assets momentum:")
    print_sorted_mom_filtered(safe, sorted_mom)
    print("Buy best safe asset on: {}$".format(round(args.amount * bf, 1)))
    top_safe = [t for t, m in sorted_mom if t in safe][-1]
    safe_target = round(args.amount * bf, 1)

    if args.amount:
        risky_top = [t for t, _ in top]
        targets = {**{top_safe: safe_target}, **{t: risky_target for t in risky_top}}

        orders = prepare_orders(read_ticker_state(args.current_fn), [top_safe] + risky_top, targets)

        for ot, ticker, amount in sorted(orders, key=lambda x: x[0]):
            val = round(amount, 1) if type(amount) != str else amount
            print("\t{}\t{}\t{}".format(ot.upper(), ticker, val))



# TBD: add descr for ticker at `buy` section yf.Ticker("VAW").info["longName"]
# TBD: move to args `risky assets`, `save assets` (or to yaml config)
if __name__ == '__main__':
    main()
