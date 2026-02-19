# utils/sentiment_analysis.py
from typing import List, Dict
import streamlit as st

FINBERT_MODEL = "yiyanghkust/finbert-tone"

_pipe = None
_load_attempted = False
_load_success = False

LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "pos": "positive",
    "neg": "negative",
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive"
}

# ── Keyword-based fallback (no ML needed) ──
POSITIVE_WORDS = [
    "gain", "gains", "up", "surge", "surges", "jump", "jumps", "rally",
    "rallies", "rise", "rises", "soar", "soars", "record", "beat", "beats",
    "positive", "bullish", "outperform", "upgrade", "upgraded", "higher",
    "boost", "boosts", "profit", "growth", "expands", "upbeat", "optimistic",
    "recover", "recovery", "strong", "strength", "buy", "overweight"
]
NEGATIVE_WORDS = [
    "loss", "losses", "down", "drop", "drops", "fall", "falls", "plunge",
    "plunges", "crash", "decline", "declines", "miss", "misses", "negative",
    "bearish", "underperform", "downgrade", "downgraded", "lower", "cut",
    "cuts", "concern", "warning", "warns", "weak", "weakness", "sell",
    "underweight", "slump", "slumps", "recession", "layoff", "layoffs",
    "debt", "default", "fraud", "scandal", "investigation", "lawsuit"
]


def _keyword_sentiment(text: str) -> Dict:
    """Simple keyword-based sentiment as fallback."""
    txt = text.lower()
    pos_count = sum(1 for w in POSITIVE_WORDS if f" {w} " in f" {txt} " or txt.startswith(w) or txt.endswith(w))
    neg_count = sum(1 for w in NEGATIVE_WORDS if f" {w} " in f" {txt} " or txt.startswith(w) or txt.endswith(w))

    if pos_count > neg_count:
        label = "positive"
        score = min(0.5 + 0.1 * pos_count, 0.95)
        numeric = score
    elif neg_count > pos_count:
        label = "negative"
        score = min(0.5 + 0.1 * neg_count, 0.95)
        numeric = -score
    else:
        label = "neutral"
        score = 0.5
        numeric = 0.0

    return {
        "label": label,
        "score": round(score, 4),
        "numeric_sentiment": round(numeric, 4),
        "influence_score": round(numeric, 4),
        "raw_label": label
    }


def get_finbert_pipeline():
    global _pipe, _load_attempted, _load_success

    if _load_attempted:
        if _load_success:
            return _pipe
        else:
            return None

    _load_attempted = True

    try:
        from transformers import BertTokenizer, BertForSequenceClassification, pipeline

        st.write("🔄 Loading FinBERT model… (first time only)")
        tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
        model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
        _pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            return_all_scores=False
        )
        _load_success = True
        return _pipe

    except Exception as e:
        st.warning(f"⚠️ FinBERT not available — using keyword-based sentiment. Reason: {e}")
        _load_success = False
        return None


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    return LABEL_MAP.get(label, "neutral")


def analyze_headlines(headlines: List[str], batch_size: int = 16) -> List[Dict]:
    if not headlines:
        return []

    pipe = get_finbert_pipeline()

    # If FinBERT failed to load, use keyword fallback
    if pipe is None:
        return [_keyword_sentiment(h) for h in headlines]

    try:
        results = pipe(headlines, batch_size=batch_size)
    except Exception as e:
        st.warning(f"⚠️ FinBERT inference failed — using keyword fallback. ({e})")
        return [_keyword_sentiment(h) for h in headlines]

    out = []
    for r in results:
        raw_label = r.get("label", "").lower()
        score = float(r.get("score", 0))
        label = normalize_label(raw_label)

        if label == "positive":
            numeric = score
        elif label == "negative":
            numeric = -score
        else:
            numeric = 0.0

        out.append({
            "label": label,
            "score": score,
            "numeric_sentiment": numeric,
            "influence_score": numeric,
            "raw_label": raw_label
        })

    return out
