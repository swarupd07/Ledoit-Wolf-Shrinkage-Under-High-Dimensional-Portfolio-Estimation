import pandas as pd
import yfinance as yf
import os

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prices_to_returns(prices):
    # Converting adjusted closing prices to returns.
    returns = prices.pct_change().dropna(how="all")
    return returns


def load_nifty500_symbols(path):
    df = pd.read_csv(path)

    symbols = df["Symbol"].dropna().unique().tolist()

    # Yahoo Finance uses .NS for NSE stocks
    symbols = [symbol + ".NS" for symbol in symbols]

    return symbols


def download_prices(symbols, start_date, end_date):
    data = yf.download(
        tickers=symbols,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=True,
        group_by="column",
        threads=True,
    )

    prices = data["Close"]

    return prices

def filter_by_coverage(prices, min_coverage=1):
    coverage = prices.notna().mean()

    eligible = coverage[coverage >= min_coverage].index

    return prices[eligible]

def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path)

    print(f"Saved: {path}")
    print(f"Shape: {df.shape}")


def load_data(path):
    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True
    )

    return df



def prices_to_returns(prices):
    returns = prices.pct_change(fill_method=None) # prevent pandas silently forward filling missing prices before calculating returns

    returns = returns.dropna(how="all")

    return returns

def select_master_universe(returns, n_assets=150, random_seed=42):
    selected = returns.sample(
        n=n_assets,
        axis=1,
        random_state=random_seed
    )

    return selected