# coding: utf-8
import yfinance as yf  # kinda docs: https://pypi.org/project/yfinance/
import pandas as pd  # https://pandas.pydata.org/docs/index.html

pd.options.mode.chained_assignment = (
    None  # default='warn' to make my excessive data copying sleep well
)

import argparse
from datetime import date
from collections import defaultdict


def MOM(last_price, sma):
    return last_price / sma - 1


def SMA(ticker_cleaned, L=12, ma_type="linear"):
    """wma: https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average"""
    if ma_type == "avg":
        weighted = pd.DataFrame(
            {"Price": ticker_cleaned.dropna()["Close"], "Weight": [1] * L},
            index=ticker_cleaned.index,
        )

    if ma_type == "linear":
        weighted = pd.DataFrame(
            {"Price": ticker_cleaned.dropna()["Close"], "Weight": range(1, L + 1)},
            index=ticker_cleaned.index,
        )

    return sum(weighted["Weight"] * weighted["Price"]) / sum(weighted["Weight"])


def get_month(row):
    return str(row["Day"])[:7]


def fetch_all_ticker_data(tickers, include_today=False):
    today = date.today()
    if today.month == 1:
        start = date(year=today.year - 2, month=12, day=1)
    else:
        start = date(year=today.year - 1, month=today.month - 1, day=1)

    if include_today:
        end = today
    else:
        end = date(year=today.year, month=today.month, day=1)

    return yf.download(
        tickers,
        start=str(start),
        end=str(end),
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,  # https://github.com/ranaroussi/yfinance/issues/266
    )


def get_ticker_info(t):
    return yf.Ticker(t).info


def get_last_prices(tickers_list):
    data = yf.download(
        tickers=tickers_list,
        period="5d",   # 5d: in case you're running script on holidays
        interval="1h",
        progress=False,  # https://github.com/ranaroussi/yfinance/issues/266
    )  
    tail = data.dropna().tail(1)["Close"]
    return {ticker: tail.iloc[0][ticker] for ticker in tail}


def cleanup_ticker_data(d, lookback):
    d = d.dropna()
    d["Day"] = d.index
    d["Mo"] = d.apply(get_month, axis=1)

    d = d.sort_values("Day").drop_duplicates(["Mo"], keep="last")
    d = d.drop(["Mo", "Open", "High", "Low", "Volume"], axis=1)
    return d.tail(lookback)


def BF(N, n, a):
    """
    calc bond fraction
    """
    n1 = a * N / 4.0
    return (N - n) / (N - n1)


def print_sorted_mom_tickers(tm_list):
    for t, mom in tm_list:
        print("{}: {}".format(t, mom))


def print_sorted_mom_filtered(tickers, all_momentum_sorted):
    for t, m in [(t, m) for t, m in all_momentum_sorted if t in set(tickers)]:
        print("{}: {}".format(t, m))


def read_ticker_state(fn):
    return pd.read_csv(fn, header=None, names=["ticker", "amount"], index_col="ticker")


def prepare_orders(ticker_current, ticker_to_sum, ticker_to_last_price):
    """
    orders to buy and to sell
    """

    orders = []
    target_tickers = ticker_to_sum.keys()
    # BUY PART
    for ticker in target_tickers:
        # print("Ticker: {}".format(ticker))
        try:
            current_shares = ticker_current.at[ticker, "amount"]
        except KeyError:
            current_shares = 0
        target_shares = ticker_to_sum[ticker] / ticker_to_last_price[ticker]
        # print("curr: {}\ttarget_shares: {}".format(current_shares, target_shares))

        amount = target_shares - current_shares
        order_type = "buy" if amount > 0 else "sell"
        orders.append((order_type, ticker, amount))

    # Sell shares that not in target but currently available
    to_sell = set(ticker_current.index) - set(target_tickers)
    for t in to_sell:
        orders.append(("sell", t, "all"))

    return orders


# TBD: use logging module and log downloaded ticker data, dates, etc.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amount",
        type=int,
        help="amount of money you want to spend for new shares",
        required=True,
    )
    parser.add_argument(
        "--current_fn", type=str, help="csv files with current positions", required=True
    )
    parser.add_argument("--risky", action="extend", nargs="+", type=str, help="space separated tickers for risky assets", required=True)
    parser.add_argument("--safe", action="extend", nargs="+", type=str, help="space separated tickers for safe assets", required=True)
    parser.add_argument(
        "--protection_range", choices=[0, 1, 2], type=int, required=True
    )
    parser.add_argument("--debug", action="store_true", default=False, required=False)
    parser.add_argument(
        "--ma_type",
        help="moving average calculation type",
        type=str,
        default="linear",
        choices=["linear", "avg"],
        required=False,
    )
    parser.add_argument("--include_today", action="store_true", default=False, required=False)
    args = parser.parse_args()
    return args


# stage1: download and cleanup data
# stage2: implement PAA algo
# stage3: calc buy/sell orders
def main():
    args = parse_args()

    risky = set(args.risky)
    safe = set(args.safe)
    curr_ticker_state = read_ticker_state(args.current_fn)
    curr = set(curr_ticker_state.index)

    all_tickers = {*risky, *safe, *curr}
    df = fetch_all_ticker_data(all_tickers, args.include_today)

    # PAA1 strategy:
    L = 13  # Lookback length (month)
    A = args.protection_range  # Protection range (0: low, 1: medium, 2: high)
    N = len(risky)  # Universum
    TOP = 6  # Number of assets in rotation
    # Paper strategy comparison: page 9

    # 
    # Calculate ticker momentum
    #
    ticker_mom = {}
    for t in all_tickers:
        try:
            ticker_data = df[t].copy(deep=True)
            cleaned_data = cleanup_ticker_data(ticker_data, L)

            sma = SMA(
                cleaned_data, L, args.ma_type
            )  # TBD: not sure if I am doing that correctly. Do I need 13 points for SMA12? Should I include p0 to SMA calculation?
            if args.debug:
                print("Ticker: {}\tsma: {}\tCleaned data:\n{}".format(t, sma, cleaned_data))
            last_price = cleaned_data.tail(1).iloc[0]["Close"]
            momentum = MOM(last_price, sma)

            if args.debug:
                print("Last price: {}\tmomentum: {}".format(last_price, momentum))

            ticker_mom[t] = momentum
        except Exception as e:
            print("ERR processing ticket '{}'\n{}".format(t, e))

    sorted_mom = sorted(ticker_mom.items(), key=lambda x: x[1])

    print("-------------------------------------")
    print("| All tickers momentum")
    print("-------------------------------------")
    print("| {:10}| {:10}| {:10}".format("is risky", "ticker", "momentum"))
    print("-------------------------------------")
    for t, mom in sorted_mom:
        extra = " " if t not in risky else "*"
        print("| {:10}| {:10}| {:10}".format(extra, t, round(mom, 3)))
    print("\n")

    # 
    # Calc other value based on ticker momentum
    #
    positive_momentum_assets = 0
    for t in risky:
        if ticker_mom[t] > 0:
            positive_momentum_assets += 1

    top = [(t, m) for t, m in sorted_mom if m > 0 and t in risky][-TOP:]

    bf = BF(N, positive_momentum_assets, A)
    risky_asset_share = (1.0 - bf) / len(
        top
    )  # Mix the risky EW portfolio with the bond part in a (1-BF)/BF fashion
    risky_target = round(args.amount * risky_asset_share, 1)


    top_safe = [t for t, m in sorted_mom if t in safe][-1]
    safe_target = args.amount * bf

    print("Important values of algorithm:")
    print("\tPositive momentum cnt for paa risky: ", positive_momentum_assets)
    print("\tShare of each risky asset: ", round(risky_asset_share, 3))
    print("\tBond fraction: ", round(bf, 3))
    print("\tBuy best safe asset on:  {}$".format(round(safe_target, 1)))
    print("\tBuy best risky asset on: {}$".format(round(args.amount - safe_target, 1)))
    print("\n")

    if args.amount:  # calc amount of shares to buy/sell
        risky_top = [t for t, _ in top]
        # targets = {**{top_safe: safe_target}, **{t: risky_target for t in risky_top}}
        targets = defaultdict(int)
        for t in risky_top:
            targets[t] += risky_target
        targets[top_safe] += safe_target
        last_price = get_last_prices(all_tickers)

        if args.debug:
            print("Ticker prices: ")
            for ticker, price in sorted(last_price.items(), key=lambda x: x[0]):
                print("\t{}\t{}".format(ticker, price))
            print("--\n")

        orders = prepare_orders(curr_ticker_state, targets, last_price)
        ticker_info = yf.Tickers(" ".join(all_tickers))
        get_info = lambda x: ticker_info.tickers[x].info.get("longName")

        print("Buy/sell instructions for amount of shares for given tickets:")
        for ot, ticker, amount in sorted(orders, key=lambda x: x[0]):
            val = round(amount, 1) if type(amount) != str else amount
            print("\t{:5}\t{:5}\t{:10}\t{}".format(ot.upper(), ticker, val, get_info(ticker)))


# TBD: check if SMA is working properly - check using TradingView or other financial charts
if __name__ == "__main__":
    main()
