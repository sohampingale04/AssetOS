"""
ASSETOS Phase 5 – Feature Engineering
Computes returns, volatility, and momentum features per stock.
Applies rolling z-score normalisation (no data leakage).
"""

import numpy as np
import pandas as pd


# ── Raw Feature Computation ───────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute 1-day, 5-day, and 20-day log returns."""
    return {
        "ret_1d":  np.log(prices / prices.shift(1)),
        "ret_5d":  np.log(prices / prices.shift(5)),
        "ret_20d": np.log(prices / prices.shift(20)),
    }


def compute_volatility(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rolling standard deviation of daily returns."""
    ret_1d = np.log(prices / prices.shift(1))
    return {
        "vol_20d": ret_1d.rolling(20).std(),
        "vol_60d": ret_1d.rolling(60).std(),
    }


def compute_momentum(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Moving averages and momentum ratio (Price / MA20)."""
    ma20 = prices.rolling(20).mean()
    ma50 = prices.rolling(50).mean()
    return {
        "ma20":     ma20,
        "ma50":     ma50,
        "momentum": prices / ma20,    # > 1 means price above MA
    }


def build_raw_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Combine all features into a flat MultiIndex DataFrame.
    Result columns: MultiIndex (feature_name, ticker)
    """
    feature_dict: dict[str, pd.DataFrame] = {}
    feature_dict.update(compute_returns(prices))
    feature_dict.update(compute_volatility(prices))
    feature_dict.update(compute_momentum(prices))

    combined = pd.concat(feature_dict, axis=1)  # columns = (feat, ticker)
    return combined


# ── Rolling Z-Score Normalisation ────────────────────────────────────────────

def rolling_zscore(
    df: pd.DataFrame,
    window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """
    Normalise each column using a rolling window mean and std.
    Avoids look-ahead bias by using only past observations.
    """
    roll_mean = df.rolling(window=window, min_periods=min_periods).mean()
    roll_std  = df.rolling(window=window, min_periods=min_periods).std()
    # Avoid division by zero
    roll_std  = roll_std.replace(0, np.nan).ffill().fillna(1.0)
    z = (df - roll_mean) / roll_std
    # Clip extreme values to keep observations bounded
    return z.clip(-3.0, 3.0)


def normalise_features(raw_features: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Apply rolling z-score normalisation to every feature column."""
    return rolling_zscore(raw_features, window=window)


# ── Final Feature Matrix ──────────────────────────────────────────────────────

FEATURE_NAMES = ["ret_1d", "ret_5d", "ret_20d", "vol_20d", "vol_60d",
                 "ma20", "ma50", "momentum"]


def build_feature_matrix(
    prices: pd.DataFrame,
    drop_initial_rows: int = 252,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full pipeline:
      1. Compute raw features
      2. Normalise with rolling z-score
      3. Drop initial warm-up rows
      4. Drop any remaining NaNs
    Returns:
        features  – DataFrame with MultiIndex columns (feat, ticker)
        tickers   – ordered list of tickers
    """
    tickers = prices.columns.tolist()

    raw  = build_raw_features(prices)
    norm = normalise_features(raw)

    # Drop warm-up period
    norm = norm.iloc[drop_initial_rows:]

    # Drop rows with any remaining NaN
    norm = norm.dropna(how="any")

    assert norm.isna().sum().sum() == 0, "NaN values remain in feature matrix!"
    print(
        f"[Features] Matrix shape: {norm.shape}  "
        f"({len(tickers)} stocks × {len(FEATURE_NAMES)} features)"
    )
    return norm, tickers


def get_feature_array(
    features: pd.DataFrame,
    tickers: list[str],
) -> np.ndarray:
    """
    Convert MultiIndex feature DataFrame → 3-D numpy array.
    Shape: (T, N, F) where T=timesteps, N=assets, F=features
    """
    n_tickers  = len(tickers)
    n_features = len(FEATURE_NAMES)
    T          = len(features)

    arr = np.zeros((T, n_tickers, n_features), dtype=np.float32)
    for f_idx, feat in enumerate(FEATURE_NAMES):
        feat_slice = features[feat][tickers].values  # (T, N)
        arr[:, :, f_idx] = feat_slice.astype(np.float32)

    return arr


def compute_asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple (not log) period returns aligned to price index."""
    return prices.pct_change().fillna(0.0)


if __name__ == "__main__":
    from data_loader import load_clean_prices, train_val_test_split

    prices         = load_clean_prices()
    train_p, val_p, test_p = train_val_test_split(prices)

    feats, tickers = build_feature_matrix(prices)
    arr            = get_feature_array(feats, tickers)
    print("Feature array shape (T, N, F):", arr.shape)
    print("Tickers:", tickers)
