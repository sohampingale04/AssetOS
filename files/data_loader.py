"""
ASSETOS Phase 5 – Data Loader
Downloads NIFTY top stocks, cleans data, and prepares raw OHLCV.
"""

import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Universe ──────────────────────────────────────────────────────────────────
NIFTY_TICKERS = [
    "RELIANCE.NS", "TCS.NS",      "HDFCBANK.NS", "INFY.NS",     "ICICIBANK.NS",
    "HINDUNILVR.NS","SBIN.NS",    "BHARTIARTL.NS","ITC.NS",     "KOTAKBANK.NS",
    "LT.NS",        "AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS",  "TITAN.NS",
    "SUNPHARMA.NS", "WIPRO.NS",   "ULTRACEMCO.NS","BAJFINANCE.NS","NESTLEIND.NS",
    "HCLTECH.NS",   "POWERGRID.NS","NTPC.NS",     "TECHM.NS",   "ONGC.NS",
]

TRAIN_END   = "2022-01-01"
VAL_END     = "2023-01-01"
START_DATE  = "2010-01-01"
END_DATE    = None          # latest available


def download_data(
    tickers: list[str] = NIFTY_TICKERS,
    start: str = START_DATE,
    end: str | None = END_DATE,
    cache_path: str = "data/raw_prices.parquet",
) -> pd.DataFrame:
    """
    Download adjusted-close + volume data for all tickers.
    Returns a MultiIndex DataFrame: (Date, Ticker) → [Adj Close, Volume]
    """
    os.makedirs("data", exist_ok=True)

    if os.path.exists(cache_path):
        print(f"[DataLoader] Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[DataLoader] Downloading {len(tickers)} tickers from {start} …")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # Keep only Close + Volume
    close  = raw["Close"].copy()
    volume = raw["Volume"].copy()

    # Stack → long format
    close_long  = close.stack(future_stack=True).rename("adj_close")
    volume_long = volume.stack(future_stack=True).rename("volume")

    df = pd.concat([close_long, volume_long], axis=1).reset_index()
    df.columns = ["date", "ticker", "adj_close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Drop rows where adj_close is NaN / zero
    df = df[df["adj_close"].notna() & (df["adj_close"] > 0)]

    df.to_parquet(cache_path)
    print(f"[DataLoader] Saved to {cache_path}. Shape: {df.shape}")
    return df


def pivot_close(df: pd.DataFrame) -> pd.DataFrame:
    """Long format → wide (Date × Ticker) adjusted-close matrix."""
    return df.pivot(index="date", columns="ticker", values="adj_close")


def remove_sparse_tickers(
    price_df: pd.DataFrame,
    min_coverage: float = 0.85,
) -> pd.DataFrame:
    """Drop tickers that have too many missing values."""
    coverage = price_df.notna().mean()
    keep = coverage[coverage >= min_coverage].index.tolist()
    dropped = set(price_df.columns) - set(keep)
    if dropped:
        print(f"[DataLoader] Dropped sparse tickers: {dropped}")
    return price_df[keep]


def forward_fill_and_check(price_df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short gaps; assert no remaining NaNs."""
    price_df = price_df.ffill(limit=5)
    price_df = price_df.dropna(how="any")
    assert price_df.isna().sum().sum() == 0, "NaNs remain after cleaning!"
    return price_df


def load_clean_prices(
    tickers: list[str] = NIFTY_TICKERS,
    start: str = START_DATE,
    end: str | None = END_DATE,
    resample_weekly: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: download → clean → (optionally) resample to weekly.
    Returns a wide DataFrame indexed by date, columns = tickers.
    """
    raw   = download_data(tickers, start, end)
    pivot = pivot_close(raw)
    pivot = remove_sparse_tickers(pivot)
    pivot = forward_fill_and_check(pivot)

    if resample_weekly:
        # Use Friday close (week-end)
        pivot = pivot.resample("W-FRI").last()
        pivot = forward_fill_and_check(pivot)
        print(f"[DataLoader] Resampled to weekly. Shape: {pivot.shape}")

    return pivot


def train_val_test_split(
    price_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = price_df.loc[:TRAIN_END].iloc[:-1]   # up to end of 2021
    val   = price_df.loc[TRAIN_END:VAL_END].iloc[1:-1]
    test  = price_df.loc[VAL_END:]
    print(
        f"[DataLoader] Split sizes → "
        f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )
    return train, val, test


if __name__ == "__main__":
    prices = load_clean_prices()
    train, val, test = train_val_test_split(prices)
    print("Tickers:", prices.columns.tolist())
    print("Date range:", prices.index[0], "→", prices.index[-1])
