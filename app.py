# app.py
# Explainable Portfolio Dashboard — Fully Dynamic Edition
# Any stock ticker works — type it in, analyze it, remove it.

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

# ======================= PAGE CONFIG ======================================
st.set_page_config(page_title="Explainable Portfolio Dashboard", layout="wide")
st.title("📊 Explainable Portfolio Dashboard")

# ================= LOAD HISTORICAL NEWS (ONCE) =================
all_news = pd.DataFrame()
LOCAL_PROJECT_ZIP = Path(__file__).resolve().parent / "data" / "historical_news.zip"
SESSION_UPLOAD_ZIP = Path("/mnt/data/historical_news.zip")
HISTORICAL_ZIP_PATH = (
    LOCAL_PROJECT_ZIP if LOCAL_PROJECT_ZIP.exists()
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

@st.cache_data
def get_local_price_panel():
    try:
        return load_price_panel()
    except Exception:
        return pd.DataFrame(columns=["ticker","date","open","high","low","close","adj_close","volume"])

# ================= SUGGESTED TICKERS (for quick-add, not locked) =================
SUGGESTED_TICKERS = {
    "🇮🇳 Indian Stocks": {
        "TCS": "TCS.NS", "Infosys": "INFY.NS", "HCL Tech": "HCLTECH.NS",
        "Wipro": "WIPRO.NS", "LTIMindtree": "LTIM.NS", "Axis Bank": "AXISBANK.NS",
        "ICICI Bank": "ICICIBANK.NS", "HDFC Bank": "HDFCBANK.NS",
        "SBI": "SBIN.NS", "Kotak Bank": "KOTAKBANK.NS", "Reliance": "RELIANCE.NS",
        "Tata Motors": "TATAMOTORS.NS", "Bharti Airtel": "BHARTIARTL.NS",
    },
    "🇺🇸 US Stocks": {
        "Apple": "AAPL", "Microsoft": "MSFT", "Google": "GOOGL",
        "Amazon": "AMZN", "Tesla": "TSLA", "NVIDIA": "NVDA",
        "Meta": "META", "Netflix": "NFLX", "JPMorgan": "JPM",
    },
    "📈 Indices & ETFs": {
        "S&P 500 ETF": "SPY", "Nasdaq ETF": "QQQ", "Nifty 50 ETF": "^NSEI",
        "Sensex": "^BSESN", "Gold ETF": "GLD", "Bitcoin ETF": "IBIT",
    },
}

# Build a flat name->ticker map for alias resolution
NAME_TO_TICKER = {}
for cat in SUGGESTED_TICKERS.values():
    for name, ticker in cat.items():
        NAME_TO_TICKER[name.upper()] = ticker
        NAME_TO_TICKER[name.split()[0].upper()] = ticker

# ================= SESSION STATE INIT =================
if "user_tickers" not in st.session_state:
    st.session_state["user_tickers"] = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "AAPL", "MSFT"]

if "portfolio_prices" not in st.session_state:
    st.session_state["portfolio_prices"] = pd.DataFrame()

# ================= HELPER FUNCTIONS =================
def normalize_ticker_guess(t: str) -> str:
    if not t:
        return ""
    t = str(t).strip().upper()
    if t in NAME_TO_TICKER:
        return NAME_TO_TICKER[t]
    return t

def fetch_portfolio_prices(tickers, start, end, batch_size=12):
    cleaned = list(dict.fromkeys([t for t in tickers if t]))
    frames, failed = [], []
    if not cleaned:
        return pd.DataFrame(), []
    for i in range(0, len(cleaned), batch_size):
        batch = cleaned[i:i+batch_size]
        try:
            df = fetch_live_prices(batch, start=start, end=end)
        except Exception:
            failed.extend(batch)
            continue
        if df is None or df.empty:
            failed.extend(batch)
            continue
        got = sorted(df['ticker'].dropna().unique().astype(str).tolist())
        failed.extend([tk for tk in batch if tk not in got])
        frames.append(df)
    prices_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return prices_df, sorted(set(failed))

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
        price_df["close"] = price_df.get("adj_close", pd.NA)
    price_df["prev_close"] = price_df.groupby("ticker")["close"].shift(1)
    price_df["daily_return"] = (price_df["close"] / price_df["prev_close"]) - 1
    price_df.drop(columns=["prev_close"], errors="ignore", inplace=True)
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


# =====================================================================
#                        SIDEBAR — TICKER MANAGEMENT
# =====================================================================
from utils.ticker_search import search_tickers

st.sidebar.header("🎯 Stock Picker")

# ── 1. Smart search bar — type company name or ticker ──
st.sidebar.markdown("**🔍 Search any stock** — type a name like `Google`, `Reliance`, `Tesla`…")
search_query = st.sidebar.text_input(
    "Search stocks",
    placeholder="Type company name or ticker…",
    key="stock_search_input"
)

# Show search results as clickable suggestions
if search_query and len(search_query.strip()) >= 1:
    suggestions = search_tickers(search_query.strip(), max_results=8)
    if suggestions:
        for s in suggestions:
            tk = s["ticker"]
            name = s.get("name", "")
            exchange = s.get("exchange", "")
            already_in = tk in st.session_state["user_tickers"]
            label = f"{'✅' if already_in else '➕'} **{tk}** — {name}"
            if exchange:
                label += f" `{exchange}`"
            if st.sidebar.button(
                label,
                key=f"search_{tk}",
                disabled=already_in,
                use_container_width=True
            ):
                if tk not in st.session_state["user_tickers"]:
                    st.session_state["user_tickers"].append(tk)
                    st.session_state["portfolio_prices"] = pd.DataFrame()
                    st.rerun()
    else:
        # No match found — offer to add raw input as ticker
        raw_ticker = search_query.strip().upper()
        st.sidebar.caption(f"No match found. Add `{raw_ticker}` directly?")
        if st.sidebar.button(f"➕ Add {raw_ticker}", key="add_raw"):
            if raw_ticker not in st.session_state["user_tickers"]:
                st.session_state["user_tickers"].append(raw_ticker)
                st.session_state["portfolio_prices"] = pd.DataFrame()
                st.rerun()

# ── 2. Quick-add from suggestions ──
st.sidebar.markdown("---")
st.sidebar.markdown("**Quick add from popular stocks:**")
for cat_name, cat_tickers in SUGGESTED_TICKERS.items():
    with st.sidebar.expander(cat_name):
        cols = st.columns(3)
        for idx, (name, ticker) in enumerate(cat_tickers.items()):
            col = cols[idx % 3]
            already_in = ticker in st.session_state["user_tickers"]
            if col.button(
                f"{'✅' if already_in else '+'} {name}",
                key=f"quick_{ticker}",
                disabled=already_in
            ):
                if ticker not in st.session_state["user_tickers"]:
                    st.session_state["user_tickers"].append(ticker)
                    st.session_state["portfolio_prices"] = pd.DataFrame()
                    st.rerun()

# ── 3. Upload portfolio CSV ──
st.sidebar.markdown("---")
st.sidebar.markdown("**Upload portfolio CSV** (optional)")
portfolio_file = st.sidebar.file_uploader("CSV with 'ticker' or 'symbol' column", type=["csv"], key="pf_upload")
if portfolio_file is not None:
    try:
        pf_df = pd.read_csv(portfolio_file)
        for col in pf_df.columns:
            if col.lower() in ["ticker", "symbol", "tickers", "code"]:
                vals = pf_df[col].dropna().astype(str).str.strip().str.upper().tolist()
                added = 0
                for v in vals:
                    if v and v not in st.session_state["user_tickers"]:
                        st.session_state["user_tickers"].append(v)
                        added += 1
                if added:
                    st.sidebar.success(f"Added {added} tickers from CSV")
                    st.session_state["portfolio_prices"] = pd.DataFrame()
                break
    except Exception as e:
        st.sidebar.error(f"Failed to parse CSV: {e}")

# ── 4. Current tickers with remove buttons ──
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Active tickers ({len(st.session_state['user_tickers'])}):**")

tickers_to_remove = []
for tk in st.session_state["user_tickers"]:
    col1, col2 = st.sidebar.columns([4, 1])
    col1.markdown(f"`{tk}`")
    if col2.button("✕", key=f"rm_{tk}"):
        tickers_to_remove.append(tk)

if tickers_to_remove:
    for tk in tickers_to_remove:
        st.session_state["user_tickers"].remove(tk)
    st.session_state["portfolio_prices"] = pd.DataFrame()
    st.rerun()

if st.sidebar.button("🗑️ Clear all tickers"):
    st.session_state["user_tickers"] = []
    st.session_state["portfolio_prices"] = pd.DataFrame()
    st.rerun()

# ── 5. News source config ──
st.sidebar.markdown("---")
st.sidebar.subheader("📰 News Sources")
use_google = st.sidebar.checkbox("Google News RSS", value=True)
use_newsapi = st.sidebar.checkbox("NewsAPI.org", value=False)
use_finnhub = st.sidebar.checkbox("Finnhub", value=False)

newsapi_key, finnhub_key = "", ""
if use_newsapi:
    newsapi_key = st.sidebar.text_input("NewsAPI key", type="password") or st.secrets.get("NEWSAPI_KEY", "")
if use_finnhub:
    finnhub_key = st.sidebar.text_input("Finnhub key", type="password") or st.secrets.get("FINNHUB_KEY", "")

active_sources = []
if use_google: active_sources.append("google_rss")
if use_newsapi and newsapi_key: active_sources.append("newsapi")
if use_finnhub and finnhub_key: active_sources.append("finnhub")

# ── Mode selection ──
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Live (yfinance + RSS)", "Historical (ZIP + local CSVs)"])

# =====================================================================
#              FINAL TICKERS — from session state
# =====================================================================
final_tickers = list(st.session_state["user_tickers"])

if not final_tickers:
    st.warning("👈 Add some tickers in the sidebar to get started!")
    st.stop()

# =====================================================================
#                      PRICE DATA FETCH
# =====================================================================
local_price_df = get_local_price_panel()
price_df_for_events = None

cached_prices = st.session_state.get("portfolio_prices", pd.DataFrame())
if not cached_prices.empty:
    cached_tickers = set(cached_prices["ticker"].dropna().unique())
    needed_tickers = set(final_tickers)
    if needed_tickers.issubset(cached_tickers):
        price_df_for_events = cached_prices.copy()

if price_df_for_events is None:
    if mode.startswith("Live"):
        st.subheader("📈 Fetching live prices…")
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=5 * 365)
            with st.spinner(f"Fetching {len(final_tickers)} tickers (5y history)..."):
                live_prices, failed = fetch_portfolio_prices(final_tickers, start=start, end=end)

            if live_prices is not None and not live_prices.empty:
                live_prices["date"] = pd.to_datetime(live_prices["date"], errors="coerce").dt.date
                live_prices["ticker"] = live_prices["ticker"].astype(str).str.upper()
                if "close" not in live_prices.columns and "adj_close" in live_prices.columns:
                    live_prices["close"] = live_prices["adj_close"]
                price_df_for_events = live_prices[
                    ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
                ].copy()
                st.session_state["portfolio_prices"] = price_df_for_events
                st.success(f"✅ Fetched {len(price_df_for_events)} rows for {price_df_for_events['ticker'].nunique()} tickers")
                if failed:
                    st.warning(f"⚠️ Could not fetch: {', '.join(failed[:20])}")
            else:
                st.warning("yfinance returned no data — falling back to local prices")
                price_df_for_events = local_price_df.copy()
        except Exception as e:
            st.error(f"Live price fetch failed: {e}")
            price_df_for_events = local_price_df.copy()
    else:
        st.subheader("📚 Historical mode — using local CSV prices")
        price_df_for_events = local_price_df.copy()

price_df_for_events["date"] = pd.to_datetime(price_df_for_events["date"], errors="coerce").dt.date

# =====================================================================
#                      STOCK PRICE CHART
# =====================================================================
st.markdown("---")
st.header("📊 Stock Price Chart")

if price_df_for_events.empty:
    st.info("No price data available yet.")
else:
    plot_df = price_df_for_events.copy()
    plot_df["date_dt"] = pd.to_datetime(plot_df["date"], errors="coerce")
    tickers_in_data = sorted(plot_df["ticker"].dropna().unique().tolist())

    selected_plot_tickers = st.multiselect(
        "Select tickers to plot", options=tickers_in_data, default=tickers_in_data
    )

    col1, col2 = st.columns([2, 1])
    with col2:
        price_col = st.selectbox("Price type", ["close", "adj_close"], index=0)
        min_d = plot_df["date_dt"].min().date()
        max_d = plot_df["date_dt"].max().date()
        date_range = st.date_input(
            "Date range", value=(min_d, max_d),
            min_value=min_d, max_value=max_d, key="price_chart_date_main"
        )
        # Reset quick range if user manually changes date picker
        if len(date_range) == 2 and st.session_state.get("chart_quick_range"):
            dr_start, dr_end = date_range
            # Check if date range matches any quick range — if not, clear it
            selected_label = st.session_state["chart_quick_range"]
            expected_start = max(min_d, max_d - time_ranges.get(selected_label, timedelta(0)))
            if dr_start != expected_start or dr_end != max_d:
                st.session_state["chart_quick_range"] = None

    # ── Quick time range buttons ──
    st.markdown("**Quick range:**")
    btn_cols = st.columns(7)
    time_ranges = {
        "1D": timedelta(days=1),
        "1W": timedelta(weeks=1),
        "1M": timedelta(days=30),
        "3M": timedelta(days=90),
        "6M": timedelta(days=180),
        "1Y": timedelta(days=365),
        "5Y": timedelta(days=5*365),
    }

    # Initialize quick range in session state
    if "chart_quick_range" not in st.session_state:
        st.session_state["chart_quick_range"] = None

    for i, (label, delta) in enumerate(time_ranges.items()):
        with btn_cols[i]:
            if st.button(label, key=f"range_{label}", use_container_width=True):
                st.session_state["chart_quick_range"] = label

    # Determine effective date range
    if st.session_state.get("chart_quick_range"):
        selected_label = st.session_state["chart_quick_range"]
        delta = time_ranges[selected_label]
        effective_end = max_d
        effective_start = max(min_d, max_d - delta)
    elif len(date_range) == 2:
        effective_start, effective_end = date_range
    else:
        effective_start, effective_end = min_d, max_d

    if selected_plot_tickers:
        mask = (
            plot_df["ticker"].isin(selected_plot_tickers)
            & (plot_df["date_dt"].dt.date >= effective_start)
            & (plot_df["date_dt"].dt.date <= effective_end)
        )
        df_plot = plot_df.loc[mask, ["date_dt", "ticker", price_col]]
        if df_plot.empty:
            st.warning("No price data in the selected date range.")
        else:
            fig = px.line(
                df_plot, x="date_dt", y=price_col, color="ticker",
                title=f"{price_col.upper()} Price Over Time",
                labels={"date_dt": "Date", price_col: "Price", "ticker": "Ticker"},
            )
            fig.update_layout(height=500, hovermode="x unified", legend_title_text="Ticker")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select at least one ticker to plot.")

# =====================================================================
#                       NEWS FETCHING
# =====================================================================
st.subheader("📰 News feed (aggregated)")

if mode.startswith("Live"):
    rss_items = []
    yesterday = date.today() - timedelta(days=1)
    one_month_ago = yesterday - timedelta(days=30)
    news_sources_to_use = active_sources if active_sources else ["google_rss"]

    with st.spinner(f"Fetching news for {len(final_tickers)} tickers…"):
        for tk in final_tickers:
            try:
                items = fetch_all_sources(
                    tk, sources=news_sources_to_use,
                    max_items_per_source=30,
                    newsapi_key=newsapi_key, finnhub_key=finnhub_key,
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
            by=["published_date", "published"], ascending=[False, False]
        ).reset_index(drop=True)
        st.success(f"Fetched {len(df_news)} headlines from {df_news['source'].nunique()} sources.")
        st.dataframe(df_news[["published","title","source","requested_ticker"]].head(100))
else:
    price_dates = pd.to_datetime(price_df_for_events["date"], errors="coerce").dt.date
    min_price_date = price_dates.min()
    max_price_date = price_dates.max()
    ticker_lookup_hist = {}
    for tk in final_tickers:
        base = tk.replace(".NS", "").replace(".BO", "").upper()
        ticker_lookup_hist[tk] = {"ticker": tk, "aliases": [tk.lower(), base.lower()]}
        ticker_lookup_hist[base] = {"ticker": tk, "aliases": [tk.lower(), base.lower()]}
    try:
        filtered_news = filter_news_by_date_and_tickers(
            all_news, list(ticker_lookup_hist.keys()),
            start_date=min_price_date, end_date=max_price_date
        )
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
        st.success(f"Filtered {len(df_news)} historical headlines")
        st.dataframe(df_news[["published","title","source"]].head(100))

# =====================================================================
#                       SENTIMENT ANALYSIS
# =====================================================================
st.subheader("🧠 Sentiment analysis")
headlines = df_news["title"].fillna("").astype(str).tolist()
if not headlines:
    st.info("No headlines to analyze.")
    st.stop()

with st.spinner("Analyzing headlines…"):
    try:
        sent = analyze_headlines(headlines)
        if not isinstance(sent, list):
            raise ValueError("analyze_headlines returned non-list")
    except Exception as e:
        st.warning(f"Sentiment analysis failed — using neutral fallback. ({e})")
        sent = [{"label":"neutral","numeric_sentiment":0.0,"score":0.5,"influence_score":0.0,"raw_label":"neutral"} for _ in headlines]

s_df = pd.DataFrame(sent)
merged = pd.concat([df_news.reset_index(drop=True), s_df.reset_index(drop=True)], axis=1)

# =====================================================================
#                       TICKER DETECTION
# =====================================================================
ticker_lookup = {}
for tk in final_tickers:
    base = tk.replace(".NS", "").replace(".BO", "").upper()
    aliases = [tk.lower(), base.lower()]
    if len(base) > 4:
        for suffix in ["BANK", "TECH", "MOTORS", "ARTL"]:
            if base.endswith(suffix):
                prefix = base[:-len(suffix)]
                if len(prefix) >= 2:
                    aliases.extend([prefix.lower(), suffix.lower()])
    ticker_lookup[tk] = {"ticker": tk, "aliases": sorted(set(aliases))}
    ticker_lookup[base] = {"ticker": tk, "aliases": sorted(set(aliases))}

prices_dir = Path(__file__).resolve().parent / "data" / "prices"
try:
    local_lookup = build_ticker_lookup(prices_dir)
    ticker_lookup.update(local_lookup)
except Exception:
    pass

merged["detected_tickers"] = (
    merged["title"].fillna("").astype(str)
    .apply(lambda t: detect_tickers_in_text(t, ticker_lookup) or [])
)

base_to_full = {}
for tk in final_tickers:
    base = tk.replace(".NS", "").replace(".BO", "").upper()
    base_to_full[base] = tk
    base_to_full[tk] = tk

# =====================================================================
#                       EVENT DATE & WINDOWS
# =====================================================================
merged["event_date"] = pd.to_datetime(merged["published"], errors="coerce").dt.date
trading_days = set(price_df_for_events["date"].dropna().unique())
merged["event_date"] = merged["event_date"].apply(lambda d: align_event_date(d, trading_days))

exploded = []
for _, r in merged.reset_index(drop=True).iterrows():
    tickers = r.get("detected_tickers") or []
    if isinstance(tickers, (list, tuple)) and len(tickers) > 0:
        for tk in tickers:
            newr = r.copy()
            newr["detected_ticker"] = base_to_full.get(str(tk).upper(), str(tk).upper())
            exploded.append(newr)

exploded_df = pd.DataFrame(exploded)

if exploded_df.empty:
    st.info("No ticker-linked events found in headlines. Try adding more tickers or checking news sources.")
    st.stop()

price_df_for_events["date"] = pd.to_datetime(price_df_for_events["date"], errors="coerce").dt.date
event_windows_df = compute_event_windows(exploded_df, price_df_for_events, backward_days=3)

# =====================================================================
#                       EVENT STUDY TABLE
# =====================================================================
st.subheader("📋 Event study (backward returns AR & CAR)")
display_cols = ["event_date","detected_ticker","title","label","numeric_sentiment","ar_-1","ar_-2","ar_-3","car_-3_-1","source","link"]
available_cols = [c for c in display_cols if c in event_windows_df.columns]
st.dataframe(
    event_windows_df[available_cols]
    .sort_values(["detected_ticker","event_date"], ascending=[True, False])
    .reset_index(drop=True).head(300)
)

# =====================================================================
#                       STOCK SELECTOR
# =====================================================================
tickers_available = sorted(event_windows_df["detected_ticker"].dropna().unique().tolist())
if tickers_available:
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Inspect single stock")
    selected_ticker = st.sidebar.selectbox("Select ticker", ["-- All --"] + tickers_available)
    if selected_ticker and selected_ticker != "-- All --":
        sel_df = event_windows_df[event_windows_df["detected_ticker"] == selected_ticker]
        sel_df = sel_df.sort_values("event_date", ascending=False).reset_index(drop=True)
        st.subheader(f"Events & impact for {selected_ticker}")
        st.dataframe(sel_df[available_cols].reset_index(drop=True))

# =====================================================================
#                       INFLUENCE RANKING
# =====================================================================
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
    top_k = st.number_input("Top K headlines", min_value=1, max_value=50, value=5, step=1)
    min_influence = st.number_input("Min influence (abs)", min_value=0.0, value=0.0, step=0.0001, format="%.6f")

ranked = event_windows_df.copy()
if min_influence > 0:
    ranked = ranked[ranked["influence_score"] >= float(min_influence)]
ranked = ranked.sort_values("influence_score", ascending=False).reset_index(drop=True)
top_df = ranked.head(int(top_k))

if top_df.empty:
    st.info("No headlines meet the influence criteria.")
else:
    disp_cols = ["event_date","detected_ticker","title","label","numeric_sentiment","car_-3_-1","influence_score","source","link"]
    disp_available = [c for c in disp_cols if c in top_df.columns]
    disp = top_df[disp_available].copy()
    for c in ["numeric_sentiment", "car_-3_-1"]:
        if c in disp.columns:
            disp[c] = disp[c].round(4)
    if "influence_score" in disp.columns:
        disp["influence_score"] = disp["influence_score"].round(6)

    st.dataframe(disp)

    st.markdown("**Top headlines (click link to open article):**")
    for _, row in disp.iterrows():
        st.markdown(
            f"- **{row.get('detected_ticker','')}** — {row.get('title','')}\n"
            f"  Influence: `{row.get('influence_score','')}`\n"
            f"  [open article]({row.get('link','')})"
        )

    csv_bytes = top_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download top headlines (CSV)", csv_bytes, "top_influencing_headlines.csv", "text/csv")
