# 🚀 Expansion & Deployment Guide

## What's New

### 1. Multi-Source News Loader (`utils/multi_news_loader.py`)
Three news sources instead of just Google RSS:

| Source | Free Tier | Signup |
|--------|-----------|--------|
| **Google News RSS** | Unlimited, no key | None needed |
| **NewsAPI.org** | 100 req/day, 1-month history | [newsapi.org/register](https://newsapi.org/register) |
| **Finnhub** | 60 calls/min, 1-year history | [finnhub.io/register](https://finnhub.io/register) |

**Usage in your existing `app.py`** — replace the RSS fetch block:

```python
from utils.multi_news_loader import fetch_all_sources

items = fetch_all_sources(
    ticker="TCS.NS",
    sources=["google_rss", "newsapi", "finnhub"],
    newsapi_key=st.secrets.get("NEWSAPI_KEY", ""),
    finnhub_key=st.secrets.get("FINNHUB_KEY", ""),
)
```

### 2. Portfolio Optimization Page (`pages/2_Portfolio_Optimization.py`)
- **Efficient frontier** via Monte-Carlo simulation
- **Max Sharpe** and **Min Volatility** optimal portfolios (using scipy)
- **Equal weight** benchmark comparison
- Weight allocation bar charts per strategy
- Strategy comparison table (return, vol, Sharpe)
- Downloadable weights CSV

### 3. News Explorer Page (`pages/3_News_Explorer.py`)
- Toggle news sources on/off
- Per-ticker headline browsing
- Source distribution pie chart
- Optional FinBERT sentiment overlay
- CSV export

---

## Project Structure After Expansion

```
your-project/
├── app.py                              # Main dashboard (your existing file)
├── pages/
│   ├── 2_Portfolio_Optimization.py     # NEW — efficient frontier
│   └── 3_News_Explorer.py             # NEW — multi-source news
├── utils/
│   ├── data_loader.py                  # Local CSV loader
│   ├── news_loader.py                  # Google RSS parser
│   ├── multi_news_loader.py            # NEW — NewsAPI + Finnhub
│   ├── sentiment_analysis.py           # FinBERT
│   ├── ticker_mapper.py                # Ticker detection
│   ├── price_fetcher.py                # yfinance wrapper
│   ├── portfolio_fetcher.py            # Batch price fetcher
│   ├── historical_news_loader.py       # ZIP news loader
│   └── plotting.py                     # Returns helper
├── modules/
│   ├── portfolio_opt.py                # Mean-variance optimizer
│   └── sentiment_analysis.py           # Alt sentiment module
├── data/
│   └── prices/                         # Your CSV files (AAPL.csv, etc.)
├── requirements.txt                    # NEW — pinned dependencies
├── .streamlit/
│   ├── config.toml                     # NEW — theme + server config
│   └── secrets.toml.example            # NEW — template for API keys
├── .gitignore                          # UPDATED
└── README.md                           # You should add one!
```

---

## Deploying to Streamlit Community Cloud

### Step 1: Prepare your GitHub repo

```bash
# In your project root
git init
git add .
git commit -m "Initial commit with expansion"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

**Important:** Make sure `data/prices/*.csv` files are committed (they're small). 
Remove `data/` and `*.csv` from `.gitignore` if needed, or add only the prices folder:

```gitignore
# Allow price CSVs
!data/prices/*.csv
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Select your repo, branch (`main`), and main file: **`app.py`**
5. Click **Deploy**

### Step 3: Add API Keys (optional)

In your deployed app's settings:
1. Click the **⋮** menu → **Settings** → **Secrets**
2. Paste:
   ```toml
   NEWSAPI_KEY = "your-newsapi-key-here"
   FINNHUB_KEY = "your-finnhub-key-here"
   ```
3. Save — the app will reboot with keys available via `st.secrets`

### Step 4: Access your pages

Streamlit automatically discovers files in `pages/`. Your sidebar will show:
- **Main** (app.py) — your existing dashboard
- **Portfolio Optimization** — efficient frontier
- **News Explorer** — multi-source headlines

---

## FinBERT on Streamlit Cloud — Memory Considerations

FinBERT + PyTorch need ~1.5 GB RAM. Streamlit Cloud free tier gives ~1 GB.

**If you hit memory limits**, add this to `requirements.txt` to use CPU-only PyTorch:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch
```

Or replace the `torch` line with:
```
torch --index-url https://download.pytorch.org/whl/cpu
```

Alternatively, create a `packages.txt` file (for apt packages if needed):
```
# packages.txt — leave empty or add system deps
```

---

## Quick Integration: Using multi_news_loader in app.py

To replace the single-source RSS logic in your main `app.py`, find the live news block and swap it:

```python
# Before (single source):
# items = news_rss_for_ticker(tk, max_items=30)

# After (multi source):
from utils.multi_news_loader import fetch_all_sources

items = fetch_all_sources(
    tk,
    sources=["google_rss"],  # add "newsapi", "finnhub" when keys available
    max_items_per_source=30,
    newsapi_key=st.secrets.get("NEWSAPI_KEY", ""),
    finnhub_key=st.secrets.get("FINNHUB_KEY", ""),
)
```

This is backward-compatible — Google RSS works without any API key.

---

## Future Expansion Ideas

Once these are working, good next steps would be:
- **Forward returns** (AR +1, +2, +3) to test if sentiment actually predicts price moves
- **Backtest engine** — simulate a sentiment-momentum strategy and plot cumulative returns
- **SHAP / attention explainability** for FinBERT predictions
- **Sector tagging** to aggregate sentiment at industry level
- **Alerts** — Slack/email notification when high-influence headlines appear
