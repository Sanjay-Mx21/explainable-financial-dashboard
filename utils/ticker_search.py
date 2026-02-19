# utils/ticker_search.py
"""
Smart ticker search — type a company name, get ticker suggestions.
Uses yfinance's search endpoint to find real tickers.
Falls back to a local dictionary for offline/fast results.
"""

import requests
from typing import List, Dict

# ── Local dictionary for instant offline suggestions ──
LOCAL_TICKER_DB = {
    # US Tech
    "apple": {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
    "microsoft": {"ticker": "MSFT", "name": "Microsoft Corp.", "exchange": "NASDAQ"},
    "google": {"ticker": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
    "alphabet": {"ticker": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
    "amazon": {"ticker": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
    "meta": {"ticker": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ"},
    "facebook": {"ticker": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ"},
    "tesla": {"ticker": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
    "nvidia": {"ticker": "NVDA", "name": "NVIDIA Corp.", "exchange": "NASDAQ"},
    "netflix": {"ticker": "NFLX", "name": "Netflix Inc.", "exchange": "NASDAQ"},
    "amd": {"ticker": "AMD", "name": "Advanced Micro Devices", "exchange": "NASDAQ"},
    "intel": {"ticker": "INTC", "name": "Intel Corp.", "exchange": "NASDAQ"},
    "adobe": {"ticker": "ADBE", "name": "Adobe Inc.", "exchange": "NASDAQ"},
    "salesforce": {"ticker": "CRM", "name": "Salesforce Inc.", "exchange": "NYSE"},
    "oracle": {"ticker": "ORCL", "name": "Oracle Corp.", "exchange": "NYSE"},
    "ibm": {"ticker": "IBM", "name": "IBM Corp.", "exchange": "NYSE"},
    "uber": {"ticker": "UBER", "name": "Uber Technologies", "exchange": "NYSE"},
    "airbnb": {"ticker": "ABNB", "name": "Airbnb Inc.", "exchange": "NASDAQ"},
    "spotify": {"ticker": "SPOT", "name": "Spotify Technology", "exchange": "NYSE"},
    "snapchat": {"ticker": "SNAP", "name": "Snap Inc.", "exchange": "NYSE"},
    "snap": {"ticker": "SNAP", "name": "Snap Inc.", "exchange": "NYSE"},
    "twitter": {"ticker": "X", "name": "X Corp. (delisted)", "exchange": "—"},
    "paypal": {"ticker": "PYPL", "name": "PayPal Holdings", "exchange": "NASDAQ"},
    "shopify": {"ticker": "SHOP", "name": "Shopify Inc.", "exchange": "NYSE"},
    "zoom": {"ticker": "ZM", "name": "Zoom Video", "exchange": "NASDAQ"},
    "palantir": {"ticker": "PLTR", "name": "Palantir Technologies", "exchange": "NYSE"},
    "coinbase": {"ticker": "COIN", "name": "Coinbase Global", "exchange": "NASDAQ"},
    "robinhood": {"ticker": "HOOD", "name": "Robinhood Markets", "exchange": "NASDAQ"},

    # US Finance
    "jpmorgan": {"ticker": "JPM", "name": "JPMorgan Chase", "exchange": "NYSE"},
    "goldman": {"ticker": "GS", "name": "Goldman Sachs", "exchange": "NYSE"},
    "morgan stanley": {"ticker": "MS", "name": "Morgan Stanley", "exchange": "NYSE"},
    "visa": {"ticker": "V", "name": "Visa Inc.", "exchange": "NYSE"},
    "mastercard": {"ticker": "MA", "name": "Mastercard Inc.", "exchange": "NYSE"},
    "bank of america": {"ticker": "BAC", "name": "Bank of America", "exchange": "NYSE"},
    "wells fargo": {"ticker": "WFC", "name": "Wells Fargo", "exchange": "NYSE"},
    "citigroup": {"ticker": "C", "name": "Citigroup Inc.", "exchange": "NYSE"},
    "american express": {"ticker": "AXP", "name": "American Express", "exchange": "NYSE"},
    "berkshire": {"ticker": "BRK-B", "name": "Berkshire Hathaway", "exchange": "NYSE"},

    # US Consumer / Industrial
    "coca cola": {"ticker": "KO", "name": "Coca-Cola Co.", "exchange": "NYSE"},
    "pepsi": {"ticker": "PEP", "name": "PepsiCo Inc.", "exchange": "NASDAQ"},
    "nike": {"ticker": "NKE", "name": "Nike Inc.", "exchange": "NYSE"},
    "disney": {"ticker": "DIS", "name": "Walt Disney Co.", "exchange": "NYSE"},
    "mcdonalds": {"ticker": "MCD", "name": "McDonald's Corp.", "exchange": "NYSE"},
    "starbucks": {"ticker": "SBUX", "name": "Starbucks Corp.", "exchange": "NASDAQ"},
    "walmart": {"ticker": "WMT", "name": "Walmart Inc.", "exchange": "NYSE"},
    "costco": {"ticker": "COST", "name": "Costco Wholesale", "exchange": "NASDAQ"},
    "boeing": {"ticker": "BA", "name": "Boeing Co.", "exchange": "NYSE"},
    "ford": {"ticker": "F", "name": "Ford Motor Co.", "exchange": "NYSE"},
    "gm": {"ticker": "GM", "name": "General Motors", "exchange": "NYSE"},
    "general motors": {"ticker": "GM", "name": "General Motors", "exchange": "NYSE"},
    "general electric": {"ticker": "GE", "name": "GE Aerospace", "exchange": "NYSE"},
    "johnson": {"ticker": "JNJ", "name": "Johnson & Johnson", "exchange": "NYSE"},
    "procter": {"ticker": "PG", "name": "Procter & Gamble", "exchange": "NYSE"},
    "exxon": {"ticker": "XOM", "name": "Exxon Mobil", "exchange": "NYSE"},
    "chevron": {"ticker": "CVX", "name": "Chevron Corp.", "exchange": "NYSE"},

    # US Healthcare
    "pfizer": {"ticker": "PFE", "name": "Pfizer Inc.", "exchange": "NYSE"},
    "moderna": {"ticker": "MRNA", "name": "Moderna Inc.", "exchange": "NASDAQ"},
    "unitedhealth": {"ticker": "UNH", "name": "UnitedHealth Group", "exchange": "NYSE"},
    "abbvie": {"ticker": "ABBV", "name": "AbbVie Inc.", "exchange": "NYSE"},

    # India — IT
    "tcs": {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "exchange": "NSE"},
    "tata consultancy": {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "exchange": "NSE"},
    "infosys": {"ticker": "INFY.NS", "name": "Infosys Ltd.", "exchange": "NSE"},
    "wipro": {"ticker": "WIPRO.NS", "name": "Wipro Ltd.", "exchange": "NSE"},
    "hcl": {"ticker": "HCLTECH.NS", "name": "HCL Technologies", "exchange": "NSE"},
    "hcl tech": {"ticker": "HCLTECH.NS", "name": "HCL Technologies", "exchange": "NSE"},
    "tech mahindra": {"ticker": "TECHM.NS", "name": "Tech Mahindra Ltd.", "exchange": "NSE"},
    "ltimindtree": {"ticker": "LTIM.NS", "name": "LTIMindtree Ltd.", "exchange": "NSE"},

    # India — Banks
    "hdfc bank": {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd.", "exchange": "NSE"},
    "hdfc": {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd.", "exchange": "NSE"},
    "icici": {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd.", "exchange": "NSE"},
    "icici bank": {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd.", "exchange": "NSE"},
    "sbi": {"ticker": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE"},
    "state bank": {"ticker": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE"},
    "kotak": {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "exchange": "NSE"},
    "kotak bank": {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "exchange": "NSE"},
    "axis": {"ticker": "AXISBANK.NS", "name": "Axis Bank Ltd.", "exchange": "NSE"},
    "axis bank": {"ticker": "AXISBANK.NS", "name": "Axis Bank Ltd.", "exchange": "NSE"},
    "indusind": {"ticker": "INDUSINDBK.NS", "name": "IndusInd Bank", "exchange": "NSE"},
    "yes bank": {"ticker": "YESBANK.NS", "name": "Yes Bank Ltd.", "exchange": "NSE"},
    "bandhan": {"ticker": "BANDHANBNK.NS", "name": "Bandhan Bank", "exchange": "NSE"},

    # India — Large Cap
    "reliance": {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "exchange": "NSE"},
    "tata motors": {"ticker": "TATAMOTORS.NS", "name": "Tata Motors Ltd.", "exchange": "NSE"},
    "tata steel": {"ticker": "TATASTEEL.NS", "name": "Tata Steel Ltd.", "exchange": "NSE"},
    "tata power": {"ticker": "TATAPOWER.NS", "name": "Tata Power Co.", "exchange": "NSE"},
    "bharti airtel": {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd.", "exchange": "NSE"},
    "airtel": {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd.", "exchange": "NSE"},
    "asian paints": {"ticker": "ASIANPAINT.NS", "name": "Asian Paints Ltd.", "exchange": "NSE"},
    "maruti": {"ticker": "MARUTI.NS", "name": "Maruti Suzuki India", "exchange": "NSE"},
    "bajaj finance": {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance Ltd.", "exchange": "NSE"},
    "bajaj finserv": {"ticker": "BAJAJFINSV.NS", "name": "Bajaj Finserv Ltd.", "exchange": "NSE"},
    "adani": {"ticker": "ADANIENT.NS", "name": "Adani Enterprises", "exchange": "NSE"},
    "adani enterprises": {"ticker": "ADANIENT.NS", "name": "Adani Enterprises", "exchange": "NSE"},
    "adani ports": {"ticker": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "exchange": "NSE"},
    "itc": {"ticker": "ITC.NS", "name": "ITC Ltd.", "exchange": "NSE"},
    "hindustan unilever": {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever", "exchange": "NSE"},
    "hul": {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever", "exchange": "NSE"},
    "larsen": {"ticker": "LT.NS", "name": "Larsen & Toubro", "exchange": "NSE"},
    "l&t": {"ticker": "LT.NS", "name": "Larsen & Toubro", "exchange": "NSE"},
    "sun pharma": {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma Industries", "exchange": "NSE"},
    "dr reddy": {"ticker": "DRREDDY.NS", "name": "Dr. Reddy's Labs", "exchange": "NSE"},
    "cipla": {"ticker": "CIPLA.NS", "name": "Cipla Ltd.", "exchange": "NSE"},
    "power grid": {"ticker": "POWERGRID.NS", "name": "Power Grid Corp.", "exchange": "NSE"},
    "ntpc": {"ticker": "NTPC.NS", "name": "NTPC Ltd.", "exchange": "NSE"},
    "ongc": {"ticker": "ONGC.NS", "name": "Oil & Natural Gas Corp.", "exchange": "NSE"},
    "coal india": {"ticker": "COALINDIA.NS", "name": "Coal India Ltd.", "exchange": "NSE"},
    "zomato": {"ticker": "ZOMATO.NS", "name": "Zomato Ltd.", "exchange": "NSE"},
    "paytm": {"ticker": "PAYTM.NS", "name": "One97 Communications", "exchange": "NSE"},
    "nykaa": {"ticker": "NYKAA.NS", "name": "FSN E-Commerce", "exchange": "NSE"},
    "dmart": {"ticker": "DMART.NS", "name": "Avenue Supermarts", "exchange": "NSE"},

    # ETFs & Indices
    "s&p 500": {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "exchange": "NYSE"},
    "spy": {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "exchange": "NYSE"},
    "nasdaq": {"ticker": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ"},
    "qqq": {"ticker": "QQQ", "name": "Invesco QQQ Trust", "exchange": "NASDAQ"},
    "nifty": {"ticker": "^NSEI", "name": "Nifty 50 Index", "exchange": "NSE"},
    "sensex": {"ticker": "^BSESN", "name": "BSE Sensex Index", "exchange": "BSE"},
    "gold": {"ticker": "GLD", "name": "SPDR Gold Shares", "exchange": "NYSE"},
    "bitcoin": {"ticker": "BTC-USD", "name": "Bitcoin USD", "exchange": "Crypto"},
    "ethereum": {"ticker": "ETH-USD", "name": "Ethereum USD", "exchange": "Crypto"},

    # UK
    "shell": {"ticker": "SHEL.L", "name": "Shell PLC", "exchange": "LSE"},
    "bp": {"ticker": "BP.L", "name": "BP PLC", "exchange": "LSE"},
    "hsbc": {"ticker": "HSBA.L", "name": "HSBC Holdings", "exchange": "LSE"},
    "barclays": {"ticker": "BARC.L", "name": "Barclays PLC", "exchange": "LSE"},
    "unilever": {"ticker": "ULVR.L", "name": "Unilever PLC", "exchange": "LSE"},

    # Japan
    "toyota": {"ticker": "7203.T", "name": "Toyota Motor Corp.", "exchange": "TSE"},
    "sony": {"ticker": "6758.T", "name": "Sony Group Corp.", "exchange": "TSE"},
    "nintendo": {"ticker": "7974.T", "name": "Nintendo Co.", "exchange": "TSE"},

    # Korea
    "samsung": {"ticker": "005930.KS", "name": "Samsung Electronics", "exchange": "KRX"},
}


def search_tickers_local(query: str, max_results: int = 8) -> List[Dict]:
    """
    Search the local dictionary for matches.
    Returns list of {ticker, name, exchange, match_key}
    """
    if not query or len(query) < 1:
        return []

    q = query.lower().strip()
    results = []
    seen_tickers = set()

    # 1. Exact key match
    if q in LOCAL_TICKER_DB:
        entry = LOCAL_TICKER_DB[q]
        if entry["ticker"] not in seen_tickers:
            results.append({**entry, "match_key": q})
            seen_tickers.add(entry["ticker"])

    # 2. Prefix match on keys
    for key, entry in LOCAL_TICKER_DB.items():
        if len(results) >= max_results:
            break
        if key.startswith(q) and entry["ticker"] not in seen_tickers:
            results.append({**entry, "match_key": key})
            seen_tickers.add(entry["ticker"])

    # 3. Substring match on keys and names
    for key, entry in LOCAL_TICKER_DB.items():
        if len(results) >= max_results:
            break
        if entry["ticker"] not in seen_tickers:
            if q in key or q in entry["name"].lower():
                results.append({**entry, "match_key": key})
                seen_tickers.add(entry["ticker"])

    # 4. Ticker symbol match (user typed the ticker directly like AAPL)
    q_upper = q.upper()
    for key, entry in LOCAL_TICKER_DB.items():
        if len(results) >= max_results:
            break
        if entry["ticker"].upper() == q_upper and entry["ticker"] not in seen_tickers:
            results.append({**entry, "match_key": key})
            seen_tickers.add(entry["ticker"])

    return results[:max_results]


def search_tickers_yfinance(query: str, max_results: int = 8) -> List[Dict]:
    """
    Search Yahoo Finance for ticker suggestions.
    This hits the public YF search API (no key needed).
    """
    if not query or len(query) < 2:
        return []

    try:
        url = "https://query2.finance.yahoo.com/v1/finance/search"
        params = {
            "q": query,
            "quotesCount": max_results,
            "newsCount": 0,
            "listsCount": 0,
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for quote in data.get("quotes", [])[:max_results]:
            ticker = quote.get("symbol", "")
            name = quote.get("shortname") or quote.get("longname") or ""
            exchange = quote.get("exchange", "")
            qtype = quote.get("quoteType", "")
            if ticker and qtype in ("EQUITY", "ETF", "INDEX", "CRYPTOCURRENCY", "MUTUALFUND", ""):
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "exchange": exchange,
                })
        return results
    except Exception:
        return []


def search_tickers(query: str, max_results: int = 8) -> List[Dict]:
    """
    Combined search: local first (instant), then Yahoo Finance (network).
    Deduplicates by ticker symbol.
    """
    local_results = search_tickers_local(query, max_results=max_results)
    yf_results = search_tickers_yfinance(query, max_results=max_results)

    seen = set()
    combined = []

    # Local results first (faster, curated)
    for r in local_results:
        tk = r["ticker"]
        if tk not in seen:
            seen.add(tk)
            r["source"] = "local"
            combined.append(r)

    # Then Yahoo Finance results
    for r in yf_results:
        tk = r["ticker"]
        if tk not in seen:
            seen.add(tk)
            r["source"] = "yahoo"
            combined.append(r)

    return combined[:max_results]
