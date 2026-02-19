# pages/2_Portfolio_Optimization.py
"""
Streamlit page: Portfolio Optimization
  • Efficient frontier (Monte-Carlo + analytical min-variance)
  • Sharpe-optimal portfolio
  • Weight allocation bar chart
  • Downloadable weights CSV
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.title("📊 Portfolio Optimization")

# ── helpers ──────────────────────────────────────────────
TRADING_DAYS = 252


def _get_price_data() -> pd.DataFrame:
    """Pull price data from session state or try live fetch."""
    if "portfolio_prices" in st.session_state and not st.session_state["portfolio_prices"].empty:
        return st.session_state["portfolio_prices"].copy()
    # fallback: try loading local CSVs
    try:
        from utils.data_loader import load_price_panel
        return load_price_panel()
    except Exception:
        return pd.DataFrame()


def _build_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot price data to daily returns matrix (columns = tickers)."""
    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    pivot = df.pivot_table(index="date", columns="ticker", values="close").sort_index()
    returns = pivot.pct_change().dropna(how="all")
    # drop tickers with too many NaNs (>30 %)
    thresh = int(len(returns) * 0.7)
    returns = returns.dropna(axis=1, thresh=thresh).dropna()
    return returns


def _monte_carlo_portfolios(mu: np.ndarray, cov: np.ndarray, n_assets: int, n_sims: int = 5000):
    results = np.zeros((n_sims, 3))  # return, vol, sharpe
    weight_arr = np.zeros((n_sims, n_assets))
    for i in range(n_sims):
        w = np.random.dirichlet(np.ones(n_assets))
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        sharpe = ret / vol if vol > 0 else 0
        results[i] = [ret, vol, sharpe]
        weight_arr[i] = w
    return results, weight_arr


def _min_variance_weights(mu, cov):
    n = len(mu)
    try:
        from scipy.optimize import minimize

        def port_vol(w):
            return np.sqrt(w @ cov @ w)

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0, 1) for _ in range(n))
        x0 = np.ones(n) / n
        res = minimize(port_vol, x0, bounds=bounds, constraints=cons, method="SLSQP")
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum()
    except ImportError:
        pass
    # fallback equal weight
    return np.ones(n) / n


def _max_sharpe_weights(mu, cov, risk_free: float = 0.0):
    n = len(mu)
    try:
        from scipy.optimize import minimize

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            return -(ret - risk_free) / vol if vol > 0 else 0

        cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = tuple((0, 1) for _ in range(n))
        x0 = np.ones(n) / n
        res = minimize(neg_sharpe, x0, bounds=bounds, constraints=cons, method="SLSQP")
        if res.success:
            w = np.maximum(res.x, 0)
            return w / w.sum()
    except ImportError:
        pass
    return np.ones(n) / n


# ── main UI ──────────────────────────────────────────────
price_df = _get_price_data()

if price_df.empty:
    st.warning(
        "No price data found. Go to the main dashboard first and fetch prices, "
        "or make sure CSVs are in `data/prices/`."
    )
    st.stop()

returns = _build_returns(price_df)
if returns.empty or returns.shape[1] < 2:
    st.warning("Need at least 2 tickers with overlapping price history for optimization.")
    st.stop()

tickers = list(returns.columns)
st.sidebar.subheader("Settings")

selected = st.sidebar.multiselect("Tickers to include", tickers, default=tickers)
risk_free = st.sidebar.number_input("Risk-free rate (annualized %)", 0.0, 20.0, 5.0, 0.5) / 100
n_sims = st.sidebar.slider("Monte-Carlo simulations", 1000, 20000, 5000, 1000)

if len(selected) < 2:
    st.info("Select at least 2 tickers.")
    st.stop()

ret_mat = returns[selected]
mu = ret_mat.mean().values * TRADING_DAYS
cov = ret_mat.cov().values * TRADING_DAYS
n = len(selected)

# ── Run simulations ──
with st.spinner("Running Monte-Carlo simulations…"):
    mc_results, mc_weights = _monte_carlo_portfolios(mu, cov, n, n_sims)

mc_df = pd.DataFrame(mc_results, columns=["Return", "Volatility", "Sharpe"])

# ── Optimal portfolios ──
w_minvol = _min_variance_weights(mu, cov)
w_sharpe = _max_sharpe_weights(mu, cov, risk_free)

minvol_ret = w_minvol @ mu
minvol_vol = np.sqrt(w_minvol @ cov @ w_minvol)

sharpe_ret = w_sharpe @ mu
sharpe_vol = np.sqrt(w_sharpe @ cov @ w_sharpe)

eq_w = np.ones(n) / n
eq_ret = eq_w @ mu
eq_vol = np.sqrt(eq_w @ cov @ eq_w)

# ── Efficient Frontier Plot ──
st.subheader("Efficient Frontier")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=mc_df["Volatility"], y=mc_df["Return"],
    mode="markers",
    marker=dict(size=3, color=mc_df["Sharpe"], colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")),
    name="Simulated",
    hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[minvol_vol], y=[minvol_ret], mode="markers+text",
    marker=dict(size=14, color="blue", symbol="diamond"),
    text=["Min Vol"], textposition="top center", name="Min Volatility",
))
fig.add_trace(go.Scatter(
    x=[sharpe_vol], y=[sharpe_ret], mode="markers+text",
    marker=dict(size=14, color="red", symbol="star"),
    text=["Max Sharpe"], textposition="top center", name="Max Sharpe",
))
fig.add_trace(go.Scatter(
    x=[eq_vol], y=[eq_ret], mode="markers+text",
    marker=dict(size=14, color="green", symbol="cross"),
    text=["Equal Wt"], textposition="top center", name="Equal Weight",
))
fig.update_layout(
    xaxis_title="Annualized Volatility",
    yaxis_title="Annualized Return",
    xaxis_tickformat=".1%", yaxis_tickformat=".1%",
    height=550, hovermode="closest",
)
st.plotly_chart(fig, use_container_width=True)

# ── Weight Allocation ──
st.subheader("Weight Allocation")

col1, col2, col3 = st.columns(3)

strategies = {
    "Max Sharpe": w_sharpe,
    "Min Volatility": w_minvol,
    "Equal Weight": eq_w,
}

for col, (name, weights) in zip([col1, col2, col3], strategies.items()):
    with col:
        st.markdown(f"**{name}**")
        w_df = pd.DataFrame({"Ticker": selected, "Weight": weights})
        w_df["Weight %"] = (w_df["Weight"] * 100).round(2)
        w_df = w_df.sort_values("Weight", ascending=False).reset_index(drop=True)

        fig_bar = px.bar(w_df, x="Ticker", y="Weight %", text="Weight %", title=name)
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(height=350, yaxis_range=[0, 100])
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Summary Table ──
st.subheader("Strategy Comparison")

summary_rows = []
for name, weights in strategies.items():
    r = weights @ mu
    v = np.sqrt(weights @ cov @ weights)
    s = (r - risk_free) / v if v > 0 else 0
    summary_rows.append({
        "Strategy": name,
        "Ann. Return": f"{r:.2%}",
        "Ann. Volatility": f"{v:.2%}",
        "Sharpe Ratio": f"{s:.2f}",
    })

st.table(pd.DataFrame(summary_rows))

# ── Download weights ──
st.subheader("Download Weights")
download_strategy = st.selectbox("Select strategy to download", list(strategies.keys()))
dl_weights = strategies[download_strategy]
dl_df = pd.DataFrame({"ticker": selected, "weight": dl_weights.round(6)})
csv_bytes = dl_df.to_csv(index=False).encode("utf-8")
st.download_button(
    f"Download {download_strategy} weights (CSV)",
    csv_bytes,
    file_name=f"{download_strategy.lower().replace(' ','_')}_weights.csv",
    mime="text/csv",
)
