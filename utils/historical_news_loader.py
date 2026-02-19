# utils/historical_news_loader.py
import zipfile
import io
import csv
import json
from pathlib import Path
from typing import List, Optional
import pandas as pd
import re

COMMON_COLUMN_MAP = {
    "published": ["published", "publish_date", "date", "datetime", "time", "pubdate"],
    "title": ["title", "headline", "head", "news_title", "article_title"],
    "summary": ["summary", "body", "description", "content", "text"],
    "link": ["link", "url", "article_url", "source_url"],
    "source": ["source", "publisher", "site", "domain"]
}

def _map_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    new_cols = {}
    for std, variants in COMMON_COLUMN_MAP.items():
        for v in variants:
            if v in cols:
                new_cols[cols[v]] = std
                break
    if new_cols:
        df = df.rename(columns=new_cols)
    for c in ["published", "title", "summary", "link", "source"]:
        if c not in df.columns:
            df[c] = pd.NA
    return df[["published", "title", "summary", "link", "source"] + [c for c in df.columns if c not in ("published","title","summary","link","source")]]

def _try_read_csv_bytes(b: bytes) -> Optional[pd.DataFrame]:
    text = None
    for enc in ("utf-8", "latin-1"):
        try:
            text = b.decode(enc)
            break
        except Exception:
            text = None
    if text is None:
        return None
    try:
        sample = text[:10000]
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        delim = dialect.delimiter
    except Exception:
        delim = ','
    try:
        df = pd.read_csv(io.StringIO(text), sep=delim, engine="python")
        return df
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(text), sep=',', engine="python")
            return df
        except Exception:
            return None

def _try_read_json_bytes(b: bytes) -> Optional[pd.DataFrame]:
    for enc in ("utf-8", "latin-1"):
        try:
            txt = b.decode(enc)
        except Exception:
            txt = None
        if not txt:
            continue
        try:
            data = json.loads(txt)
            if isinstance(data, dict):
                if "articles" in data and isinstance(data["articles"], list):
                    data = data["articles"]
                elif "data" in data and isinstance(data["data"], list):
                    data = data["data"]
                else:
                    data = [data]
            if isinstance(data, list):
                return pd.DataFrame(data)
        except Exception:
            try:
                rows = [json.loads(line) for line in txt.splitlines() if line.strip()]
                return pd.DataFrame(rows)
            except Exception:
                continue
    return None

def load_historical_news_from_zip(zip_path: str | Path) -> pd.DataFrame:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    all_frames = []
    with zipfile.ZipFile(zip_path, "r") as z:
        file_list = [n for n in z.namelist() if n.lower().endswith((".csv", ".json"))]
        for name in file_list:
            try:
                with z.open(name) as f:
                    raw = f.read()
                df = None
                if name.lower().endswith(".csv"):
                    df = _try_read_csv_bytes(raw)
                elif name.lower().endswith(".json"):
                    df = _try_read_json_bytes(raw)
                if df is None:
                    continue
                df = _map_columns(df)
                if "published" in df.columns:
                    df["published"] = df["published"].astype("string")
                df["__source_file"] = name
                all_frames.append(df)
            except Exception:
                continue

    if not all_frames:
        cols = ["published", "title", "summary", "link", "source", "__source_file"]
        return pd.DataFrame(columns=cols)

    merged = pd.concat(all_frames, ignore_index=True, sort=False)

    if "title" in merged.columns:
        merged["title"] = merged["title"].astype("string").str.strip()

    merged["published_parsed"] = pd.to_datetime(merged["published"], errors="coerce", infer_datetime_format=True)
    merged["published"] = merged["published_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S")
    merged.loc[merged["published"].isna(), "published"] = merged.loc[merged["published"].isna(), "published_parsed"].astype(str)
    merged.drop(columns=["published_parsed"], inplace=True, errors=True)

    if "title" in merged.columns and "published" in merged.columns:
        merged = merged.drop_duplicates(subset=["title", "published"], keep="first").reset_index(drop=True)

    return merged

def filter_news_by_date_and_tickers(news_df: pd.DataFrame,
                                    tickers: Optional[List[str]] = None,
                                    start_date=None,
                                    end_date=None) -> pd.DataFrame:
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=news_df.columns if news_df is not None else ["published","title","summary","link","source"])

    df = news_df.copy()
    df["published_dt"] = pd.to_datetime(df["published"], errors="coerce")
    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        df = df[df["published_dt"] >= start_ts]
    if end_date is not None:
        end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[df["published_dt"] <= end_ts]

    if tickers:
        escaped = [re.escape(t.upper()) for t in tickers if isinstance(t, str)]
        if escaped:
            pat = r"(?i)\b(" + "|".join(escaped) + r")\b"
            mask = df["title"].fillna("").astype(str).str.contains(pat, regex=True) | df["summary"].fillna("").astype(str).str.contains(pat, regex=True)
            df = df[mask]

    df.drop(columns=["published_dt"], inplace=True, errors=True)
    return df.reset_index(drop=True)
