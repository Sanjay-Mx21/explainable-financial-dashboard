# pages/3_News_Explorer.py
"""
Streamlit page: Multi-Source News Explorer
  • Toggle between Google RSS, NewsAPI, Finnhub
  • Per-ticker headline view with sentiment
  • Source distribution chart
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="News Explorer", layout="wide")
st.title("📰 Multi-Source News Explorer")

# ── imports ──
try:
    from utils.multi_news_loader import fetch_all_sources
except ImportError:
    st.error("Could not import `utils.multi_news_loader`. Make sure the file exists.")
    st.stop()

try:
    from utils.sentiment_analysis import analyze_headlines
    HAS_SENTIMENT = True
except Exception:
    HAS_SENTIMENT = False

# ── sidebar config ──
st.sidebar.subheader("News Sources")
use_google = st.sidebar.checkbox("Google News RSS", value=True)
use_newsapi = st.sidebar.checkbox("NewsAPI.org", value=False)
use_finnhub = st.sidebar.checkbox("Finnhub", value=False)

newsapi_key = ""
finnhub_key = ""
if use_newsapi:
    newsapi_key = st.sidebar.text_input("NewsAPI key", type="password",
                                         help="Get free key at https://newsapi.org/register")
if use_finnhub:
    finnhub_key = st.sidebar.text_input("Finnhub key", type="password",
                                         help="Get free key at https://finnhub.io/register")

st.sidebar.markdown("---")
ticker_input = st.sidebar.text_input("Ticker(s) — comma separated", value="TCS.NS, INFY.NS, HDFCBANK.NS")
max_per_source = st.sidebar.slider("Max headlines per source", 5, 50, 20)

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if not tickers:
    st.info("Enter at least one ticker in the sidebar.")
    st.stop()

# ── build source list ──
sources = []
if use_google:
    sources.append("google_rss")
if use_newsapi and newsapi_key:
    sources.append("newsapi")
if use_finnhub and finnhub_key:
    sources.append("finnhub")

if not sources:
    st.warning("Enable at least one news source in the sidebar.")
    st.stop()

# ── fetch ──
all_items = []
with st.spinner(f"Fetching news for {len(tickers)} tickers from {len(sources)} source(s)…"):
    for tk in tickers:
        items = fetch_all_sources(
            tk,
            sources=sources,
            max_items_per_source=max_per_source,
            newsapi_key=newsapi_key,
            finnhub_key=finnhub_key,
        )
        all_items.extend(items)

if not all_items:
    st.warning("No headlines found. Try different tickers or enable more sources.")
    st.stop()

df = pd.DataFrame(all_items)
df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)
df = df.dropna(subset=["published"]).sort_values("published", ascending=False).reset_index(drop=True)
df["published"] = df["published"].dt.tz_localize(None)

st.success(f"Fetched **{len(df)}** unique headlines across {df['source'].nunique()} sources.")

# ── optional sentiment ──
if HAS_SENTIMENT:
    run_sent = st.checkbox("Run FinBERT sentiment on headlines", value=False)
    if run_sent:
        with st.spinner("Analyzing sentiment…"):
            try:
                results = analyze_headlines(df["title"].fillna("").tolist())
                s_df = pd.DataFrame(results)
                df = pd.concat([df.reset_index(drop=True), s_df.reset_index(drop=True)], axis=1)
            except Exception as e:
                st.warning(f"Sentiment analysis failed: {e}")

# ── source distribution ──
st.subheader("Source Distribution")
import plotly.express as px

source_counts = df["source"].value_counts().reset_index()
source_counts.columns = ["Source", "Count"]
fig = px.pie(source_counts, names="Source", values="Count", title="Headlines by Source")
st.plotly_chart(fig, use_container_width=True)

# ── headlines table ──
st.subheader("Headlines")

# per-ticker filter
filter_ticker = st.selectbox("Filter by ticker", ["All"] + tickers)
view_df = df if filter_ticker == "All" else df[df["requested_ticker"] == filter_ticker]

display_cols = ["published", "title", "source", "requested_ticker"]
if "label" in view_df.columns:
    display_cols.append("label")
if "numeric_sentiment" in view_df.columns:
    display_cols.append("numeric_sentiment")

st.dataframe(view_df[display_cols].head(200), use_container_width=True)

# ── download ──
csv_bytes = view_df.to_csv(index=False).encode("utf-8")
st.download_button("Download headlines CSV", csv_bytes, "headlines_multi_source.csv", "text/csv")
