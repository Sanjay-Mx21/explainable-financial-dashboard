# app.py
# Explainable Portfolio Dashboard — Live / Historical Mode (with price chart)

import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta, date
import pandas as pd
from urllib.parse import quote_plus
import dateutil.parser
import traceback
import plotly.express as px

# utils
from utils.data_loader import load_price_panel
from utils.news_loader import parse_rss_feed
from utils.sentiment_analysis import analyze_headlines
from utils.ticker_mapper import build_ticker_lookup, detect_tickers_in_text
from utils.price_fetcher import fetch_live_prices
from utils.historical_news_loader import load_historical_news_from_zip, filter_news_by_date_and_tickers
from utils.multi_news_loader import fetch_all_sources

# ================= LOAD HISTORICAL NEWS (ONCE) =================
all_news = pd.DataFrame()

LOCAL_PROJECT_ZIP = Path(__file__).resolve().parent / "data" / "historical_news.zip"
SESSION_UPLOAD_ZIP = Path("/mnt/data/historical_news.zip")
HISTORICAL_ZIP_PATH = (
    LOCAL_PROJECT_ZIP
    if LOCAL_PROJECT_ZIP.exists()
    else (SESSION_UPLOAD_ZIP if SESSION_UPLOAD_ZIP.exists() else None)
)

if HISTORICAL_ZIP_PATH is not None:
    try:
        all_news = load_historical_news_from_zip(HISTORICAL_ZIP_PATH)
        if not all_news.empty:
            all_news["published"] = pd.to_datetime(all_news["published"], errors="coerce")
            all_news = all_news.dropna(subset=["published"])
    except Exception as e:
        st.warning(f"Failed to load historical news ZIP: {e}")
        all_news = pd.DataFrame()


# ======================= streamlit setup ======================================
st.set_page_config(page_title="Explainable Portfolio Dashboard", layout="wide")
st.title("Explainable Portfolio Dashboard — Live / Historical Mode")

# Quick link to uploaded project ZIP (local path)
ZIP_PATH = "/mnt/data/explainable_portfolio_dashboard.zip"
if Path(ZIP_PATH).exists():
    st.markdown(f"**Project ZIP:** [{ZIP_PATH}]({ZIP_PATH})")

# Prefer local ZIP if present (for historical/backtest)
LOCAL_PROJECT_ZIP = Path(__file__).resolve().parent / "data" / "historical_news.zip"
SESSION_UPLOAD_ZIP = Path("/mnt/data/historical_news.zip")
HISTORICAL_ZIP_PATH = LOCAL_PROJECT_ZIP if LOCAL_PROJECT_ZIP.exists() else (SESSION_UPLOAD_ZIP if SESSION_UPLOAD_ZIP.exists() else None)

@st.cache_data
def get_local_price_panel():
    return load_price_panel()

# ================= FIXED STOCK UNIVERSE =================
FIXED_TICKERS = {
    "TCS": "TCS.NS",
    "INFOSYS": "INFY.NS",
    "HCL": "HCLTECH.NS",
    "WIPRO": "WIPRO.NS",
    "LTIMINDTREE": "LTIM.NS",
    "AXIS": "AXISBANK.NS",
    "ICICI": "ICICIBANK.NS",
    "HDFC": "HDFCBANK.NS",
    "SBI": "SBIN.NS",
    "KOTAK": "KOTAKBANK.NS"
}

FINAL_TICKERS = list(FIXED_TICKERS.values())


# ----------------- Helper functions -----------------
COMMON_TICKER_COLS = ["ticker", "symbol", "tickers", "code", "symbolid"]

def _extract_tickers_from_df(df: pd.DataFrame):
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

def normalize_ticker_guess(t: str, default_append_ns: bool = False):
    if not t:
        return ""
    t = str(t).strip().upper()
    if "." in t:
        return t
    if t.isalpha() and len(t) <= 5:
        return t
    if default_append_ns and t.isalpha():
        return f"{t}.NS"
    return t

def fetch_portfolio_prices(tickers, start: datetime, end: datetime, batch_size: int = 12, default_append_ns: bool = False):
    cleaned = []
    for t in tickers:
        n = normalize_ticker_guess(t, default_append_ns=default_append_ns)
        if n:
            cleaned.append(n)
    seen = set()
    cleaned_unique = []
    for x in cleaned:
        if x not in seen:
            seen.add(x); cleaned_unique.append(x)
    frames = []
    failed = []
    if not cleaned_unique:
        return pd.DataFrame(), []
    for i in range(0, len(cleaned_unique), batch_size):
        batch = cleaned_unique[i:i+batch_size]
        try:
            df = fetch_live_prices(batch, start=start, end=end)
        except Exception as e:
            failed.extend(batch)
            st.sidebar.warning(f"fetch_live_prices exception for batch {batch}: {e}")
            continue
        if df is None or df.empty:
            failed.extend(batch)
            continue
        got = sorted(df['ticker'].dropna().unique().astype(str).tolist())
        missing = [tk for tk in batch if tk not in got]
        failed.extend(missing)
        frames.append(df)
    prices_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    failed_unique = sorted(list(set(failed)))
    return prices_df, failed_unique

def _normalize_rss_item(item: dict) -> dict:
    title = item.get("title") or item.get("headline") or ""
    summary = item.get("summary") or item.get("description") or item.get("content") or ""
    link = item.get("link") or item.get("url") or ""
    source = item.get("source") or item.get("publisher") or item.get("feed") or ""
    pub = item.get("published") or item.get("pubDate") or item.get("published_parsed") or item.get("date")
    pub_dt = None
    if isinstance(pub, str):
        try:
            pub_dt = dateutil.parser.parse(pub)
        except Exception:
            pub_dt = None
    elif hasattr(pub, "tm_year"):
        try:
            pub_dt = datetime(*pub[:6])
        except Exception:
            pub_dt = None
    elif isinstance(pub, (datetime, pd.Timestamp)):
        pub_dt = pd.to_datetime(pub)
    else:
        pub_dt = None
    return {"title": title, "summary": summary, "link": link, "source": source, "published": pub_dt}

def news_rss_for_ticker(ticker, max_items=30):
    q = quote_plus(f"{ticker} stock")
    rss_url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        raw_items = parse_rss_feed(rss_url, max_items=max_items) or []
    except Exception:
        raw_items = []
    out = []
    for it in raw_items:
        norm = _normalize_rss_item(it)
        norm["requested_ticker"] = str(ticker).upper()
        out.append(norm)
    return out

# ----------------- compute_event_windows -----------------
def align_event_date(event_date, trading_dates):
    trading_dates = sorted(trading_dates)
    if not trading_dates:
        return None
    if event_date > trading_dates[-1]:
        return trading_dates[-1]
    for d in trading_dates:
        if d >= event_date:
            return d
    return None


def compute_event_windows(exploded_events_df, price_df, backward_days=3):
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce").dt.date
    price_df = price_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "close" not in price_df.columns:
        if "adj_close" in price_df.columns:
            price_df["close"] = price_df["adj_close"]
        else:
            price_df["close"] = pd.NA

    price_df["prev_close"] = price_df.groupby("ticker")["close"].shift(1)
    price_df["daily_return"] = (price_df["close"] / price_df["prev_close"]) - 1
    price_df = price_df.drop(columns=["prev_close"], errors="ignore")

    ticker_groups = {t: g.set_index("date").sort_index() for t, g in price_df.groupby("ticker")}

    out = exploded_events_df.copy().reset_index(drop=True)

    for d in range(1, backward_days + 1):
        out[f"ar_-{d}"] = None
    out["car_-3_-1"] = None

    for i, row in out.iterrows():
        t = row.get("detected_ticker")
        if not t or pd.isna(t):
            continue

        t = str(t).upper()
        if t not in ticker_groups:
            continue

        price_ts = ticker_groups[t]
        trading_dates = list(price_ts.index)

        event_date = row.get("event_date")
        if pd.isna(event_date):
            continue

        if isinstance(event_date, (pd.Timestamp, datetime)):
            event_date = event_date.date()

        if event_date not in trading_dates:
            continue

        event_idx = trading_dates.index(event_date)

        ar_vals = []
        for d in range(1, backward_days + 1):
            idx = event_idx - d
            if idx >= 0:
                td = trading_dates[idx]
                ret = price_ts.at[td, "daily_return"]
                ret = None if pd.isna(ret) else float(ret)
            else:
                ret = None

            out.at[i, f"ar_-{d}"] = ret
            ar_vals.append(ret)

        numeric_vals = [v for v in ar_vals if v is not None]
        out.at[i, "car_-3_-1"] = sum(numeric_vals) if numeric_vals else None

    return out

# ----------------- UI: Mode and sidebar inputs -----------------
mode = st.sidebar.radio("Mode", options=["Live (yfinance + RSS)", "Historical (ZIP + local CSVs)"])
st.sidebar.markdown("Tip: Live mode fetches recent prices (5y) & recent news. Historical mode uses uploaded ZIP/local CSVs for backtest.")

# ── News source config in sidebar ──
st.sidebar.markdown("---")
st.sidebar.subheader("📰 News Sources")
use_google = st.sidebar.checkbox("Google News RSS", value=True)
use_newsapi = st.sidebar.checkbox("NewsAPI.org", value=False)
use_finnhub = st.sidebar.checkbox("Finnhub", value=False)

newsapi_key = ""
finnhub_key = ""
if use_newsapi:
    newsapi_key = st.sidebar.text_input("NewsAPI key", type="password",
                                         help="Get free key at https://newsapi.org/register")
    if not newsapi_key:
        newsapi_key = st.secrets.get("NEWSAPI_KEY", "")
if use_finnhub:
    finnhub_key = st.sidebar.text_input("Finnhub key", type="password",
                                         help="Get free key at https://finnhub.io/register")
    if not finnhub_key:
        finnhub_key = st.secrets.get("FINNHUB_KEY", "")

# Build active sources list
active_sources = []
if use_google:
    active_sources.append("google_rss")
if use_newsapi and newsapi_key:
    active_sources.append("newsapi")
if use_finnhub and finnhub_key:
    active_sources.append("finnhub")

# load local price panel (if available)
try:
    local_price_df = get_local_price_panel()
except Exception:
    local_price_df = pd.DataFrame(columns=["ticker","date","open","high","low","close","adj_close","volume"])

# Build ticker lookup from data/prices
prices_dir = Path(__file__).resolve().parent / "data" / "prices"
try:
    ticker_lookup = build_ticker_lookup(prices_dir)
    sanitized_lookup = {}
    if isinstance(ticker_lookup, dict):
        for k, v in ticker_lookup.items():
            kk = str(k).upper()
            if "_" in kk and not kk.endswith(".NS"):
                cand = kk.split("_")[0]
                sanitized_lookup[cand] = v
            sanitized_lookup[kk] = v
        ticker_lookup = sanitized_lookup or ticker_lookup
except Exception:
    ticker_lookup = {}

available_tickers = sorted(list(ticker_lookup.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Select tickers to analyze / Upload portfolio")
st.sidebar.write("Local detected tickers: " + (", ".join(available_tickers[:50]) if available_tickers else "none"))

selected_tickers = st.sidebar.multiselect("Tickers (detected from data/prices)", options=available_tickers, default=available_tickers[:3])

st.sidebar.markdown("---")
st.sidebar.subheader("Upload portfolio CSV (optional)")
portfolio_file = st.sidebar.file_uploader("CSV with 'ticker' or 'symbol' column", type=["csv"])
portfolio_tickers = []
if portfolio_file is not None:
    try:
        pf_df = pd.read_csv(portfolio_file)
        portfolio_tickers = _extract_tickers_from_df(pf_df)
        st.sidebar.markdown(f"Detected {len(portfolio_tickers)} tickers from uploaded file (showing first 20): {portfolio_tickers[:20]}")
    except Exception as e:
        st.sidebar.error(f"Failed to parse portfolio CSV: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("Paste tickers (comma/newline separated)")
manual_list = st.sidebar.text_area("Paste tickers (e.g. AAPL, TCS.NS)", value="", height=80)
manual_parsed = []
if manual_list and manual_list.strip():
    manual_parsed = [p.strip().upper() for p in manual_list.replace("\r","\n").replace(",", "\n").split("\n") if p.strip()]
    st.sidebar.markdown(f"Parsed {len(manual_parsed)} tickers from paste (first 20): {manual_parsed[:20]}")

st.sidebar.markdown("---")
st.sidebar.subheader("Quick add a ticker")
quick_ticker = st.sidebar.text_input("Add a single ticker (e.g. AAPL or TCS.NS)", value="")
if quick_ticker and quick_ticker.strip():
    q = quick_ticker.strip().upper()
    if q not in selected_tickers:
        selected_tickers.append(q)

# 🔒 Override: use fixed stock universe only
final_tickers = FINAL_TICKERS.copy()
st.sidebar.markdown("### 📌 Fixed Stock Universe")
for k, v in FIXED_TICKERS.items():
    st.sidebar.markdown(f"- {k} ({v})")
st.sidebar.markdown(f"**Final tickers to use:** {len(final_tickers)} (showing first 30): {final_tickers[:30]}")

default_append_ns = st.sidebar.checkbox("Append .NS to bare long tickers (India)", value=False)

# Button to fetch portfolio prices
st.sidebar.markdown("---")
if st.sidebar.button("Fetch portfolio prices (live)"):
    if not final_tickers:
        st.sidebar.error("No tickers provided.")
    else:
        st.sidebar.info(f"Fetching {len(final_tickers)} tickers in batches...")
        end = datetime.utcnow()
        start = end - timedelta(days=5 * 365)
        try:
            prices_df, failed = fetch_portfolio_prices(final_tickers, start=start, end=end, batch_size=12, default_append_ns=default_append_ns)
            if prices_df is None or prices_df.empty:
                st.sidebar.warning("No price rows returned.")
                st.session_state["portfolio_prices"] = pd.DataFrame()
            else:
                prices_df["date"] = pd.to_datetime(prices_df["date"], errors="coerce").dt.tz_localize(None).dt.date
                prices_df["ticker"] = prices_df["ticker"].astype(str).str.upper()
                st.session_state["portfolio_prices"] = prices_df
                st.sidebar.success(f"Fetched {len(prices_df)} price rows for {len(prices_df['ticker'].unique())} tickers.")
                st.dataframe(prices_df.groupby("ticker").size().reset_index(name="rows").sort_values("rows", ascending=False))
            if failed:
                st.sidebar.warning(f"No data for: {failed[:50]}")
        except Exception as e:
            st.sidebar.error(f"Portfolio fetch failed: {e}")
            st.sidebar.error(traceback.format_exc())

# ================= PRICE DATA FETCH =================
price_df_for_events = None

if (
    "portfolio_prices" in st.session_state
    and isinstance(st.session_state["portfolio_prices"], pd.DataFrame)
    and not st.session_state["portfolio_prices"].empty
):
    price_df_for_events = st.session_state["portfolio_prices"].copy()
    st.info(f"Using portfolio prices from session ({len(price_df_for_events)} rows)")

else:
    if mode.startswith("Live"):
        st.subheader("📈 Live price fetch (yfinance)")
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=5 * 365)

            tickers_to_fetch = final_tickers if final_tickers else selected_tickers

            if not tickers_to_fetch:
                st.warning("No tickers provided to fetch prices.")
                price_df_for_events = local_price_df.copy()
            else:
                with st.spinner(f"Fetching {len(tickers_to_fetch)} tickers (5y history)..."):
                    live_prices = fetch_live_prices(tickers_to_fetch, start=start, end=end)

                if live_prices is not None and not live_prices.empty:
                    live_prices["date"] = pd.to_datetime(
                        live_prices["date"], errors="coerce"
                    ).dt.date
                    live_prices["ticker"] = live_prices["ticker"].astype(str).str.upper()

                    if "close" not in live_prices.columns and "adj_close" in live_prices.columns:
                        live_prices["close"] = live_prices["adj_close"]

                    price_df_for_events = live_prices[
                        ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
                    ].copy()

                    st.success(
                        f"Fetched {len(price_df_for_events)} rows "
                        f"for {price_df_for_events['ticker'].nunique()} tickers"
                    )
                else:
                    st.warning("yfinance returned no data — falling back to local prices")
                    price_df_for_events = local_price_df.copy()

        except Exception as e:
            st.error(f"Live price fetch failed: {e}")
            st.error(traceback.format_exc())
            price_df_for_events = local_price_df.copy()

    else:
        st.subheader("📚 Historical mode — using local CSV prices")
        price_df_for_events = local_price_df.copy()

# Normalize date column
price_df_for_events["date"] = pd.to_datetime(
    price_df_for_events["date"], errors="coerce"
).dt.date

# ================= STOCK PRICE CHART =================
st.markdown("---")
st.header("📊 Stock Price Chart")

if price_df_for_events is None or price_df_for_events.empty:
    st.info("No price data available yet.")
else:
    plot_df = price_df_for_events.copy()
    plot_df["date_dt"] = pd.to_datetime(plot_df["date"], errors="coerce")

    tickers_available = sorted(plot_df["ticker"].dropna().unique().tolist())
    default_sel = tickers_available

    selected_plot_tickers = st.multiselect(
        "Select tickers to plot",
        options=tickers_available,
        default=default_sel
    )

    col1, col2 = st.columns([2, 1])
    with col2:
        price_col = st.selectbox("Price type", ["close", "adj_close"], index=0)

        min_d = plot_df["date_dt"].min().date()
        max_d = plot_df["date_dt"].max().date()

        date_range = st.date_input(
            "Date range",
            value=(min_d, max_d),
            min_value=min_d,
            max_value=max_d,
            key="price_chart_date_main"
        )

    if selected_plot_tickers:
        start_d, end_d = date_range
        mask = (
            plot_df["ticker"].isin(selected_plot_tickers)
            & (plot_df["date_dt"].dt.date >= start_d)
            & (plot_df["date_dt"].dt.date <= end_d)
        )

        df_plot = plot_df.loc[mask, ["date_dt", "ticker", price_col]]

        if df_plot.empty:
            st.warning("No price data in the selected date range.")
        else:
            fig = px.line(
                df_plot,
                x="date_dt",
                y=price_col,
                color="ticker",
                title=f"{price_col.upper()} Price Over Time",
                labels={"date_dt": "Date", price_col: "Price", "ticker": "Ticker"},
            )
            fig.update_layout(
                height=500,
                hovermode="x unified",
                legend_title_text="Ticker"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one ticker to plot.")

# ----------------- NEWS FETCHING (MULTI-SOURCE) -----------------
st.subheader("📰 News feed (aggregated)")
if 'price_df_for_events' not in locals() or price_df_for_events is None:
    price_df_for_events = pd.DataFrame(columns=["ticker","date","close"])

if mode.startswith("Live"):
    rss_items = []

    yesterday = date.today() - timedelta(days=1)
    one_month_ago = yesterday - timedelta(days=30)

    tickers_to_query = final_tickers if final_tickers else selected_tickers

    # ✅ Use multi-source loader
    news_sources_to_use = active_sources if active_sources else ["google_rss"]

    for tk in tickers_to_query:
        try:
            items = fetch_all_sources(
                tk,
                sources=news_sources_to_use,
                max_items_per_source=30,
                newsapi_key=newsapi_key,
                finnhub_key=finnhub_key,
            )
            for it in items:
                it["requested_ticker"] = tk.upper()
                pub = it.get("published")
                if isinstance(pub, (datetime, pd.Timestamp)):
                    pub_date = pub.date()
                    if one_month_ago <= pub_date <= yesterday:
                        rss_items.append(it)
        except Exception:
            continue

    if not rss_items:
        st.warning("No news fetched for selected tickers.")
        df_news = pd.DataFrame(columns=["published","title","summary","link","source","requested_ticker"])
    else:
        df_news = pd.DataFrame(rss_items)
        df_news["published"] = pd.to_datetime(df_news["published"], errors="coerce")
        df_news = df_news.dropna(subset=["published"])
        df_news["published"] = df_news["published"].dt.tz_localize(None)
        df_news["published_date"] = df_news["published"].dt.date
        df_news = df_news.sort_values(
            by=["published_date", "published"],
            ascending=[False, False]
        ).reset_index(drop=True)

        src_counts = df_news["source"].value_counts()
        st.success(f"Fetched {len(df_news)} headlines from {len(src_counts)} sources across selected tickers.")
        st.dataframe(
            df_news[["published","title","source","requested_ticker"]].head(100)
        )

else:
    price_dates = pd.to_datetime(price_df_for_events["date"], errors="coerce").dt.date
    min_price_date = price_dates.min()
    max_price_date = price_dates.max()
    prices_dir = Path(__file__).resolve().parent / "data" / "prices"
    ticker_lookup = build_ticker_lookup(prices_dir)
    for name, ticker in FIXED_TICKERS.items():
        ticker_lookup[name] = {
            "ticker": ticker,
            "aliases": [name, name.replace("BANK", ""), name.title()]
        }

    available_tickers = sorted(list(ticker_lookup.keys()))
    try:
        filtered_news = filter_news_by_date_and_tickers(all_news, available_tickers, start_date=min_price_date, end_date=max_price_date)
    except Exception:
        filtered_news = pd.DataFrame()
    if filtered_news.empty:
        st.warning("No historical news overlapping price date range.")
        df_news = pd.DataFrame(columns=["published","title","summary","link","source"])
    else:
        df_news = filtered_news.copy()
        df_news["published"] = pd.to_datetime(df_news["published"], errors="coerce")
        df_news = df_news.dropna(subset=["published"]).reset_index(drop=True)
        df_news["published"] = df_news["published"].dt.tz_localize(None)
        df_news = df_news.sort_values("published", ascending=False).reset_index(drop=True)
        st.success(f"Filtered {len(df_news)} historical headlines for date range {min_price_date} → {max_price_date}")
        st.dataframe(df_news[["published","title","source"]].head(100))

# ----------------- SENTIMENT -----------------
st.subheader("🧠 Sentiment analysis")
headlines = df_news["title"].fillna("").astype(str).tolist()
if not headlines:
    st.info("No headlines to analyze.")
    st.stop()

with st.spinner("Analyzing headlines (FinBERT or neutral fallback)..."):
    try:
        sent = analyze_headlines(headlines)
        if not isinstance(sent, list):
            raise ValueError("analyze_headlines returned non-list")
    except Exception as e:
        st.warning(f"analyze_headlines failed — using neutral fallback. ({e})")
        sent = [{"label":"neutral","numeric_sentiment":0.0} for _ in headlines]

s_df = pd.DataFrame(sent)
merged = pd.concat([df_news.reset_index(drop=True), s_df.reset_index(drop=True)], axis=1)

# ----------------- TICKER DETECTION -----------------
prices_dir = Path(__file__).resolve().parent / "data" / "prices"
ticker_lookup = build_ticker_lookup(prices_dir)

for name, ticker in FIXED_TICKERS.items():
    ticker_lookup[name] = {
        "ticker": ticker,
        "aliases": [
            name,
            ticker.replace(".NS", ""),
            ticker.replace(".NS", " BANK"),
            ticker.replace(".NS", "BANK"),
            name.title()
        ]
    }

merged["detected_tickers"] = (
    merged["title"]
    .fillna("")
    .astype(str)
    .apply(lambda t: detect_tickers_in_text(t, ticker_lookup) or [])
)

# ----------------- EVENT DATE -----------------
merged["event_date"] = pd.to_datetime(
    merged["published"], errors="coerce"
).dt.date

trading_days = set(price_df_for_events["date"].dropna().unique())

merged["event_date"] = merged["event_date"].apply(
    lambda d: align_event_date(d, trading_days)
)

# ----------------- EXPLODE & EVENT WINDOWS -----------------
exploded = []
for _, r in merged.reset_index(drop=True).iterrows():
    tickers = r.get("detected_tickers") or []
    if isinstance(tickers, (list, tuple)) and len(tickers) > 0:
        for tk in tickers:
            newr = r.copy()
            newr["detected_ticker"] = tk
            exploded.append(newr)
exploded_df = pd.DataFrame(exploded)

if exploded_df.empty:
    st.info("No detected ticker events available for event study.")
    st.stop()

if price_df_for_events is None or price_df_for_events.empty:
    price_df_for_events = local_price_df.copy()
price_df_for_events["date"] = pd.to_datetime(price_df_for_events["date"], errors="coerce").dt.date

exploded_df = exploded_df.copy()
exploded_df["detected_ticker"] = (
    exploded_df["detected_ticker"]
    .astype(str)
    .str.upper()
    .map(lambda x: FIXED_TICKERS.get(x, x))
)

event_windows_df = compute_event_windows(exploded_df, price_df_for_events, backward_days=3)

st.subheader("Event study (forward returns 1..3 days and CAR 1-3)")
st.dataframe(event_windows_df[[
    "event_date","detected_ticker","title","label","numeric_sentiment","ar_-1","ar_-2","ar_-3","car_-3_-1","source","link"
]].sort_values(["detected_ticker","event_date"], ascending=[True, False]).reset_index(drop=True).head(300))

# ----------------- STOCK SELECTOR & RANKING -----------------
tickers_available = sorted(event_windows_df["detected_ticker"].dropna().unique().tolist())
if tickers_available:
    st.sidebar.subheader("Portfolio / Stock selector")
    selected_ticker = st.sidebar.selectbox("Select ticker to inspect", ["-- All --"] + tickers_available)
    if selected_ticker and selected_ticker != "-- All --":
        sel_df = event_windows_df[event_windows_df["detected_ticker"] == selected_ticker].sort_values("event_date", ascending=False).reset_index(drop=True)
        st.subheader(f"Events & impact for {selected_ticker}")
        st.dataframe(sel_df[["event_date","title","label","numeric_sentiment","ar_-1","ar_-2","ar_-3","car_-3_-1","source","link"]].reset_index(drop=True))
    else:
        st.subheader("Events (all tickers)")
        st.dataframe(event_windows_df[["event_date","detected_ticker","title","numeric_sentiment","ar_-1","ar_-2","ar_-3","car_-3_-1","source","link"]].sort_values(["detected_ticker","event_date"], ascending=[True, False]).reset_index(drop=True))
else:
    st.info("No tickers in event windows yet.")

# ================= INFLUENCE RANKING =================
st.subheader("🏆 Top influencing headlines (|sentiment × backward CAR|)")

event_windows_df["numeric_sentiment"] = pd.to_numeric(
    event_windows_df["numeric_sentiment"], errors="coerce"
).fillna(0.0)

event_windows_df["car_-3_-1"] = pd.to_numeric(
    event_windows_df["car_-3_-1"], errors="coerce"
).fillna(0.0)

event_windows_df["influence_score"] = (
    event_windows_df["numeric_sentiment"].abs()
    * event_windows_df["car_-3_-1"].abs()
)

col1, col2 = st.columns([2, 1])
with col2:
    top_k = st.number_input(
        "Top K headlines", min_value=1, max_value=50, value=5, step=1
    )
    min_influence = st.number_input(
        "Min influence (abs)",
        min_value=0.0,
        value=0.0,
        step=0.0001,
        format="%.6f"
    )

ranked = event_windows_df.copy()
if min_influence > 0:
    ranked = ranked[ranked["influence_score"] >= float(min_influence)]

ranked = ranked.sort_values("influence_score", ascending=False).reset_index(drop=True)
top_df = ranked.head(int(top_k))

if top_df.empty:
    st.info("No headlines meet the influence criteria.")
else:
    disp = top_df[[
        "event_date",
        "detected_ticker",
        "title",
        "label",
        "numeric_sentiment",
        "car_-3_-1",
        "influence_score",
        "source",
        "link"
    ]].copy()

    disp["numeric_sentiment"] = disp["numeric_sentiment"].round(4)
    disp["car_-3_-1"] = disp["car_-3_-1"].round(4)
    disp["influence_score"] = disp["influence_score"].round(6)

    st.dataframe(disp)

    st.markdown("**Top headlines (click link to open article):**")
    for _, row in disp.iterrows():
        st.markdown(
            f"- **{row['detected_ticker']}** — {row['title']}  \n"
            f"  Influence: `{row['influence_score']}`  \n"
            f"  [open article]({row.get('link','')})"
        )

    csv_bytes = top_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download top headlines (CSV)",
        csv_bytes,
        file_name="top_influencing_headlines.csv",
        mime="text/csv"
    )
