# utils/ticker_mapper.py
from pathlib import Path
import pandas as pd
import re
from difflib import get_close_matches
from typing import Dict, List, Set

def build_ticker_lookup(prices_dir: Path) -> Dict[str, Dict]:
    lookup = {}
    prices_dir = Path(prices_dir)
    for csv_path in sorted(prices_dir.glob("*.csv")):
        ticker = csv_path.stem.upper()
        name = None
        try:
            df = pd.read_csv(csv_path, nrows=5)
            cols = [c.lower().strip() for c in df.columns]
            if "name" in cols:
                name = df.iloc[0][[c for c in df.columns if c.lower().strip()=="name"][0]]
            elif "company" in cols:
                name = df.iloc[0][[c for c in df.columns if c.lower().strip()=="company"][0]]
        except Exception:
            name = None

        if pd.isna(name) or name is None:
            name = None
        else:
            name = str(name).strip()

        aliases = set()
        aliases.add(ticker.lower())
        if name:
            aliases.add(name.lower())
            for token in re.findall(r"[A-Za-z0-9]+", name.lower()):
                if len(token) > 2:
                    aliases.add(token)

        lookup[ticker] = {"ticker": ticker, "name": name, "aliases": sorted(list(aliases))}
    return lookup

def safe_detect_tickers(text, ticker_lookup):
    safe_lookup = {}
    for k, v in ticker_lookup.items():
        if isinstance(v, dict):
            safe_lookup[k] = v
        elif isinstance(v, str):
            safe_lookup[k] = {"aliases": [v.replace("_DATA", ""), v]}
    try:
        return detect_tickers_in_text(text, safe_lookup)
    except Exception:
        return []

def detect_tickers_in_text(text: str, lookup: Dict[str, Dict]) -> List[str]:
    if not text or not lookup:
        return []

    text_lower = text.lower()
    found: Set[str] = set()

    for match in re.findall(r"\$([A-Za-z]{1,6})", text):
        t = match.upper()
        if t in lookup:
            found.add(t)

    for t, meta in lookup.items():
        pattern = r"\b" + re.escape(t.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.add(t)

    for t, meta in lookup.items():
        for alias in meta.get("aliases", []):
            if alias and alias in text_lower:
                found.add(t)
                break
        if t in found:
            continue
        name = meta.get("name")
        if name and name.lower() in text_lower:
            found.add(t)

    if not found:
        alias_map = {}
        alias_list = []
        for t, meta in lookup.items():
            for a in meta.get("aliases", []):
                alias_map[a] = t
                alias_list.append(a)
        tokens = re.findall(r"[A-Za-z0-9]{3,}", text_lower)
        for tok in tokens:
            matches = get_close_matches(tok, alias_list, n=3, cutoff=0.85)
            for m in matches:
                found.add(alias_map.get(m))

    return sorted([t for t in found if t])

def explode_news_by_ticker(df_news, lookup):
    rows = []
    for _, row in df_news.reset_index(drop=True).iterrows():
        title = str(row.get("title", ""))
        detected = detect_tickers_in_text(title, lookup)
        if detected:
            for t in detected:
                new = row.to_dict()
                new["detected_ticker"] = t
                rows.append(new)
        else:
            new = row.to_dict()
            new["detected_ticker"] = None
            rows.append(new)
    return pd.DataFrame(rows)
