# utils/price_fetcher.py
from datetime import datetime
import pandas as pd
import yfinance as yf
from typing import Union, Iterable
import numpy as np

def _normalize_yf_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        date_col = None
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                date_col = c
                break
        if date_col is not None:
            if date_col != "date":
                df = df.rename(columns={date_col: "date"})
        else:
            df["date"] = pd.to_datetime(df.index)
    else:
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date"] = pd.to_datetime(pd.Series([pd.NaT] * len(df)))

    cols_lower = {c.lower(): c for c in df.columns}

    def _pick(*names):
        for n in names:
            if n.lower() in cols_lower:
                return cols_lower[n.lower()]
        return None

    open_col = _pick("open", "o")
    high_col = _pick("high", "h")
    low_col  = _pick("low", "l")
    adj_col  = _pick("adj close", "adj_close", "adjclose", "adjusted close", "adjusted_close")
    close_col = _pick("close", "c") or adj_col
    volume_col = _pick("volume", "v")

    safe = pd.DataFrame()
    safe["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)

    for name, col in (("open", open_col), ("high", high_col), ("low", low_col),
                      ("close", close_col), ("adj_close", adj_col), ("volume", volume_col)):
        if col is not None:
            safe[name] = pd.to_numeric(df[col], errors="coerce")
        else:
            safe[name] = pd.NA

    safe["ticker"] = str(ticker).upper()
    out = safe.loc[:, ["date","open","high","low","close","adj_close","volume","ticker"]]
    return out

def fetch_live_prices(tickers: Union[str, Iterable[str]], start: datetime=None, end: datetime=None, period: str=None) -> pd.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]

    tickers = [t.upper() for t in tickers]

    try:
        tickers_param = " ".join(tickers)
        if period and (start is None and end is None):
            raw = yf.download(tickers_param, period=period, group_by='ticker', threads=True, progress=False, auto_adjust=False)
        else:
            start_s = None if start is None else start.strftime("%Y-%m-%d")
            end_s = None if end is None else end.strftime("%Y-%m-%d")
            raw = yf.download(tickers_param, start=start_s, end=end_s, group_by='ticker', threads=True, progress=False, auto_adjust=False)
    except Exception:
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume","ticker"])

    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume","ticker"])

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        for tk in tickers:
            tk_upper = tk.upper()
            try:
                sub = raw.xs(tk_upper, axis=1, level=1)
                if not sub.empty:
                    frames.append(_normalize_yf_df(sub, tk_upper))
                    continue
            except Exception:
                pass

            mask_cols = [col for col in raw.columns if any(tk_upper == str(x).upper() for x in col if pd.notna(x))]
            if mask_cols:
                sub = raw.loc[:, mask_cols].copy()
                newcols = []
                for col in sub.columns:
                    cands = [str(x) for x in col if pd.notna(x)]
                    chosen = None
                    for cand in cands:
                        if cand.upper() != tk_upper:
                            chosen = cand
                            break
                    if chosen is None:
                        chosen = cands[-1]
                    newcols.append(chosen)
                sub.columns = newcols
                frames.append(_normalize_yf_df(sub, tk_upper))
            else:
                continue
    else:
        if len(tickers) == 1:
            frames.append(_normalize_yf_df(raw, tickers[0]))
        else:
            raw_cols = [str(c).upper() for c in raw.columns]
            for tk in tickers:
                tk_upper = tk.upper()
                cols_for_t = [orig for orig, up in zip(raw.columns, raw_cols) if tk_upper in up]
                if cols_for_t:
                    sub = raw.loc[:, cols_for_t].copy()
                    frames.append(_normalize_yf_df(sub, tk_upper))
                else:
                    continue

    if not frames:
        try:
            f = _normalize_yf_df(raw, tickers[0])
            frames.append(f)
        except Exception:
            return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume","ticker"])

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
    out = out.sort_values(["ticker","date"]).reset_index(drop=True)
    return out
