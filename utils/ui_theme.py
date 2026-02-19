# utils/ui_theme.py
"""
Dark Modern UI Theme for Explainable Portfolio Dashboard
Injects custom CSS for: cards, sidebar, animations, typography, spacing
"""

import streamlit as st


def inject_custom_css():
    """Call this once at the top of app.py after set_page_config"""
    st.markdown("""
    <style>
    /* ═══════════════════════════════════════════
       IMPORTS — Google Fonts
    ═══════════════════════════════════════════ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ═══════════════════════════════════════════
       GLOBAL
    ═══════════════════════════════════════════ */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Smooth scrolling */
    html { scroll-behavior: smooth; }

    /* ═══════════════════════════════════════════
       HEADER / TITLE
    ═══════════════════════════════════════════ */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.5px !important;
        background: linear-gradient(135deg, #00D2FF 0%, #7B61FF 50%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding-bottom: 0.5rem !important;
        font-size: 2.2rem !important;
    }

    h2 {
        font-weight: 600 !important;
        color: #E0E0E0 !important;
        letter-spacing: -0.3px !important;
        border-bottom: 2px solid rgba(123, 97, 255, 0.3);
        padding-bottom: 0.5rem !important;
        margin-top: 2rem !important;
    }

    h3 {
        font-weight: 500 !important;
        color: #CCCCCC !important;
    }

    /* ═══════════════════════════════════════════
       SIDEBAR
    ═══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #161B22 100%) !important;
        border-right: 1px solid rgba(123, 97, 255, 0.15) !important;
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        background: none !important;
        -webkit-text-fill-color: #E0E0E0 !important;
        font-size: 1.1rem !important;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.88rem;
        color: #9CA3AF;
    }

    /* Sidebar text input */
    [data-testid="stSidebar"] input[type="text"] {
        background: #1A1F2E !important;
        border: 1px solid rgba(123, 97, 255, 0.25) !important;
        border-radius: 10px !important;
        color: #E0E0E0 !important;
        padding: 0.6rem 0.8rem !important;
        font-size: 0.9rem !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }

    [data-testid="stSidebar"] input[type="text"]:focus {
        border-color: #7B61FF !important;
        box-shadow: 0 0 0 3px rgba(123, 97, 255, 0.15) !important;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] button {
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
    }

    [data-testid="stSidebar"] button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(123, 97, 255, 0.2) !important;
    }

    /* ═══════════════════════════════════════════
       BUTTONS (main area)
    ═══════════════════════════════════════════ */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1.2rem !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(123, 97, 255, 0.3) !important;
        background: rgba(123, 97, 255, 0.08) !important;
    }

    .stButton > button:hover {
        background: rgba(123, 97, 255, 0.2) !important;
        border-color: #7B61FF !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(123, 97, 255, 0.25) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ═══════════════════════════════════════════
       DATA TABLES
    ═══════════════════════════════════════════ */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(123, 97, 255, 0.15) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* ═══════════════════════════════════════════
       METRICS / KPI CARDS
    ═══════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(123, 97, 255, 0.08) 0%, rgba(0, 210, 255, 0.05) 100%) !important;
        border: 1px solid rgba(123, 97, 255, 0.15) !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(123, 97, 255, 0.15);
    }

    [data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 1.6rem !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ═══════════════════════════════════════════
       ALERTS / STATUS MESSAGES
    ═══════════════════════════════════════════ */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        animation: slideIn 0.4s ease-out;
    }

    /* Success */
    [data-testid="stAlert"][data-baseweb*="positive"],
    div[data-testid="stSuccess"] {
        background: rgba(16, 185, 129, 0.1) !important;
        border-left: 4px solid #10B981 !important;
    }

    /* Warning */
    div[data-testid="stWarning"] {
        background: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid #F59E0B !important;
    }

    /* Error */
    div[data-testid="stError"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid #EF4444 !important;
    }

    /* Info */
    div[data-testid="stInfo"] {
        background: rgba(123, 97, 255, 0.1) !important;
        border-left: 4px solid #7B61FF !important;
    }

    /* ═══════════════════════════════════════════
       MULTISELECT / SELECTBOX
    ═══════════════════════════════════════════ */
    [data-testid="stMultiSelect"] > div,
    [data-testid="stSelectbox"] > div {
        border-radius: 10px !important;
    }

    /* Selected chips in multiselect */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: linear-gradient(135deg, #7B61FF, #00D2FF) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        border: none !important;
    }

    /* ═══════════════════════════════════════════
       TABS
    ═══════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(123, 97, 255, 0.15) !important;
        border-color: #7B61FF !important;
    }

    /* ═══════════════════════════════════════════
       RADIO BUTTONS (horizontal)
    ═══════════════════════════════════════════ */
    .stRadio > div {
        gap: 0.5rem !important;
    }

    .stRadio > div > label {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
        padding: 0.4rem 1rem !important;
        transition: all 0.2s ease !important;
        font-size: 0.88rem !important;
    }

    .stRadio > div > label:hover {
        border-color: rgba(123, 97, 255, 0.4) !important;
        background: rgba(123, 97, 255, 0.08) !important;
    }

    /* ═══════════════════════════════════════════
       EXPANDERS
    ═══════════════════════════════════════════ */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.02) !important;
        transition: border-color 0.2s ease !important;
    }

    [data-testid="stExpander"]:hover {
        border-color: rgba(123, 97, 255, 0.25) !important;
    }

    /* ═══════════════════════════════════════════
       SPINNER / LOADING
    ═══════════════════════════════════════════ */
    .stSpinner > div {
        border-color: #7B61FF transparent transparent transparent !important;
    }

    /* ═══════════════════════════════════════════
       PLOTLY CHARTS — border & padding
    ═══════════════════════════════════════════ */
    [data-testid="stPlotlyChart"] {
        border: 1px solid rgba(123, 97, 255, 0.1);
        border-radius: 14px;
        padding: 0.5rem;
        background: rgba(255,255,255,0.01);
    }

    /* ═══════════════════════════════════════════
       DOWNLOAD BUTTON
    ═══════════════════════════════════════════ */
    [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #7B61FF 0%, #00D2FF 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }

    [data-testid="stDownloadButton"] > button:hover {
        opacity: 0.9 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(123, 97, 255, 0.3) !important;
    }

    /* ═══════════════════════════════════════════
       DIVIDERS
    ═══════════════════════════════════════════ */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(123, 97, 255, 0.3), transparent) !important;
        margin: 2rem 0 !important;
    }

    /* ═══════════════════════════════════════════
       ANIMATIONS
    ═══════════════════════════════════════════ */
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 5px rgba(123, 97, 255, 0.1); }
        50% { box-shadow: 0 0 20px rgba(123, 97, 255, 0.2); }
    }

    /* Animate main content sections on load */
    .main .block-container > div > div {
        animation: fadeIn 0.5s ease-out;
    }

    /* ═══════════════════════════════════════════
       SCROLLBAR (Webkit)
    ═══════════════════════════════════════════ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0D1117;
    }

    ::-webkit-scrollbar-thumb {
        background: rgba(123, 97, 255, 0.3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: rgba(123, 97, 255, 0.5);
    }

    /* ═══════════════════════════════════════════
       FILE UPLOADER
    ═══════════════════════════════════════════ */
    [data-testid="stFileUploader"] {
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] > div {
        border: 2px dashed rgba(123, 97, 255, 0.2) !important;
        border-radius: 12px !important;
        transition: border-color 0.3s ease !important;
    }

    [data-testid="stFileUploader"] > div:hover {
        border-color: rgba(123, 97, 255, 0.5) !important;
    }

    /* ═══════════════════════════════════════════
       NUMBER INPUT
    ═══════════════════════════════════════════ */
    [data-testid="stNumberInput"] input {
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)


def render_metric_card(label, value, delta=None, delta_color="normal"):
    """Render a styled metric using st.metric (CSS handles the styling)"""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_section_header(icon, title, subtitle=None):
    """Render a styled section header with icon"""
    st.markdown(f"## {icon} {title}")
    if subtitle:
        st.caption(subtitle)


def render_kpi_row(metrics: list):
    """
    Render a row of KPI metrics.
    metrics: list of dicts with keys: label, value, delta (optional)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(
                label=m["label"],
                value=m["value"],
                delta=m.get("delta"),
                delta_color=m.get("delta_color", "normal")
            )
