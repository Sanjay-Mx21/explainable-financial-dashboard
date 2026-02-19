# Explainable Portfolio Dashboard

🔗 **Live Demo:** [Click here](https://explainable-financial-dashboard-psqjm2ztlgetjgmukcscwy.streamlit.app/)

A Streamlit-based dashboard that combines stock price analysis with news sentiment (FinBERT) to create an explainable event study framework.

## Features

- **Live & Historical modes** — fetch prices via yfinance or use local CSVs
- **Multi-source news** — Google RSS, NewsAPI, Finnhub
- **FinBERT sentiment analysis** on headlines
- **Event study** — backward abnormal returns (AR) and cumulative AR (CAR)
- **Influence ranking** — top headlines ranked by |sentiment × CAR|
- **Portfolio optimization** — efficient frontier, Max Sharpe, Min Volatility
- **Multi-source news explorer** — compare headlines across providers

## Quick Start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
```
├── app.py                          # Main dashboard
├── pages/
│   ├── 2_Portfolio_Optimization.py # Efficient frontier & weights
│   └── 3_News_Explorer.py         # Multi-source news browser
├── utils/                          # Core utilities
├── modules/                        # Optimization & sentiment modules
├── data/prices/                    # Your stock CSV files go here
├── requirements.txt
└── .streamlit/config.toml
```

## Deployment

See [EXPANSION_GUIDE.md](EXPANSION_GUIDE.md) for full Streamlit Cloud deployment instructions.

## Data

Place your stock CSVs in `data/prices/` with columns: `date, open, high, low, close, volume`.
Files should be named by ticker: `AAPL.csv`, `TCS.csv`, etc.
