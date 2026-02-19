# utils/multi_news_loader.py
"""
Multi-source news aggregator.
Supports:
  1. Google News RSS (free, no key needed)
  2. NewsAPI.org  (free tier: 100 req/day — get key at https://newsapi.org)
  3. Finnhub      (free tier: 60 calls/min — get key at https://finnhub.io)

Each source returns a list[dict] with canonical keys:
  published (datetime | None), title, summary, link, source, requested_ticker
"""

import os
import time
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote_plus


# ──────────────────────────────────────────────
# 1.  Google News RSS  (always available)
# ──────────────────────────────────────────────
def fetch_google_rss(ticker: str, max_items: int = 30) -> List[Dict]:
    q = quote_plus(f"{ticker} stock")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        pub = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            pub = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)
        items.append({
            "published": pub,
            "title": getattr(e, "title", ""),
            "summary": getattr(e, "summary", ""),
            "link": getattr(e, "link", ""),
            "source": "Google News RSS",
            "requested_ticker": ticker.upper(),
        })
    return items


# ──────────────────────────────────────────────
# 2.  NewsAPI.org
# ──────────────────────────────────────────────
def fetch_newsapi(
    ticker: str,
    api_key: Optional[str] = None,
    max_items: int = 30,
    lookback_days: int = 30,
) -> List[Dict]:
    """
    Fetch headlines from NewsAPI 'everything' endpoint.
    Set NEWSAPI_KEY env var or pass api_key directly.
    Free plan limits: 100 requests/day, 1-month history.
    """
    key = api_key or os.getenv("NEWSAPI_KEY", "")
    if not key:
        return []

    from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} stock",
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": min(max_items, 100),
        "language": "en",
        "apiKey": key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    items = []
    for art in data.get("articles", [])[:max_items]:
        pub = None
        if art.get("publishedAt"):
            try:
                pub = datetime.fromisoformat(art["publishedAt"].replace("Z", "+00:00"))
            except Exception:
                pass
        items.append({
            "published": pub,
            "title": art.get("title", ""),
            "summary": art.get("description", ""),
            "link": art.get("url", ""),
            "source": art.get("source", {}).get("name", "NewsAPI"),
            "requested_ticker": ticker.upper(),
        })
    return items


# ──────────────────────────────────────────────
# 3.  Finnhub  (company news endpoint)
# ──────────────────────────────────────────────
def fetch_finnhub(
    ticker: str,
    api_key: Optional[str] = None,
    max_items: int = 30,
    lookback_days: int = 30,
) -> List[Dict]:
    """
    Finnhub /company-news endpoint.
    Set FINNHUB_KEY env var or pass api_key.
    Free plan: 60 API calls/min.
    Note: Finnhub uses US tickers (strip .NS etc. for Indian stocks).
    """
    key = api_key or os.getenv("FINNHUB_KEY", "")
    if not key:
        return []

    # Finnhub expects plain US tickers; strip exchange suffixes for best results
    clean_ticker = ticker.split(".")[0].upper()

    to_date = datetime.utcnow().strftime("%Y-%m-%d")
    from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": clean_ticker,
        "from": from_date,
        "to": to_date,
        "token": key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    items = []
    for art in data[:max_items]:
        pub = None
        ts = art.get("datetime")
        if ts:
            try:
                pub = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            except Exception:
                pass
        items.append({
            "published": pub,
            "title": art.get("headline", ""),
            "summary": art.get("summary", ""),
            "link": art.get("url", ""),
            "source": art.get("source", "Finnhub"),
            "requested_ticker": ticker.upper(),
        })
    return items


# ──────────────────────────────────────────────
# Unified aggregator
# ──────────────────────────────────────────────
def fetch_all_sources(
    ticker: str,
    sources: Optional[List[str]] = None,
    max_items_per_source: int = 30,
    newsapi_key: Optional[str] = None,
    finnhub_key: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch news from multiple sources and merge.
    sources: list of enabled source names, e.g. ["google_rss", "newsapi", "finnhub"]
             Default: all available (keys permitting).
    Returns de-duplicated list sorted newest-first.
    """
    if sources is None:
        sources = ["google_rss", "newsapi", "finnhub"]

    all_items: List[Dict] = []

    if "google_rss" in sources:
        all_items.extend(fetch_google_rss(ticker, max_items=max_items_per_source))

    if "newsapi" in sources:
        all_items.extend(fetch_newsapi(ticker, api_key=newsapi_key, max_items=max_items_per_source))

    if "finnhub" in sources:
        all_items.extend(fetch_finnhub(ticker, api_key=finnhub_key, max_items=max_items_per_source))

    # De-duplicate by title (case-insensitive)
    seen_titles = set()
    unique = []
    for it in all_items:
        key = (it.get("title") or "").strip().lower()
        if key and key not in seen_titles:
            seen_titles.add(key)
            unique.append(it)

    # Sort newest first
    unique.sort(key=lambda x: x.get("published") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    return unique
