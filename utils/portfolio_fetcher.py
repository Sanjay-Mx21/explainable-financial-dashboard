# utils/portfolio_fetcher.py
import pandas as pd
from typing import List, Tuple
from datetime import datetime
from utils.price_fetcher import fetch_live_prices
import streamlit as st

COMMON_TICKER_COLS = ["ticker", "symbol", "tickers", "code", "symbolid"]

def _extract_tickers_from_df(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    df = df.copy()
    for col in df.columns:
        if col.lower() in COMMON_TICKER_COLS:
            vals = df[col].dropna().astype(str).str.strip().tolist()
            return [v.upper() for v in vals if v.strip()]
    first_col = df.columns[0]
    vals = df[first_col].dropna().astype(str).str.strip().tolist()
    return [v.upper() for v in vals if v.strip()]

def normalize_ticker_guess(t: str, default_append_ns: bool = False) -> str:
    t = str(t).strip().upper()
    if not t:
        return t
    if "." in t:
        return t
    if t.isalpha() and len(t) <= 5:
        return t
    if default_append_ns and t.isalpha():
        return f"{t}.NS"
    return t

def fetch_portfolio_prices(tickers: List[str], start: datetime, end: datetime,
                           batch_size: int = 12, default_append_ns: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    cleaned = []
    for t in tickers:
        t2 = normalize_ticker_guess(t, default_append_ns=default_append_ns)
        if t2:
            cleaned.append(t2)
    cleaned = list(dict.fromkeys(cleaned))

    prices_frames = []
    failed = []
    if not cleaned:
        return pd.DataFrame(), []

    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i+batch_size]
        try:
            df = fetch_live_prices(batch, start=start, end=end)
        except Exception as e:
            st.warning(f"fetch_live_prices raised for batch {batch}: {e}")
            failed.extend(batch)
            continue

        if df is None or df.empty:
            failed.extend(batch)
            continue

        got = sorted(df["ticker"].dropna().unique().astype(str).tolist())
        missing = [tk for tk in batch if tk not in got]
        failed.extend(missing)
        prices_frames.append(df)

    prices_df = pd.concat(prices_frames, ignore_index=True, sort=False) if prices_frames else pd.DataFrame()
    failed_unique = sorted(list(set(failed)))
    return prices_df, failed_unique
