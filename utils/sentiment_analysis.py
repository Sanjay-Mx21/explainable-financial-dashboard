# utils/sentiment_analysis.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict
import streamlit as st

FINBERT_MODEL = "yiyanghkust/finbert-tone"

_tokenizer = None
_model = None
_pipe = None

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

def get_finbert_pipeline():
    global _tokenizer, _model, _pipe
    if _pipe is None:
        try:
            st.write("🔄 Loading FinBERT model… (first time only)")
            _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
            _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
            _pipe = pipeline(
                "sentiment-analysis",
                model=_model,
                tokenizer=_tokenizer,
                truncation=True,
                return_all_scores=False
            )
        except Exception as e:
            st.error(f"❌ FinBERT failed to load: {e}")
            raise
    return _pipe


def normalize_label(label: str) -> str:
    label = label.lower().strip()
    return LABEL_MAP.get(label, "neutral")


def analyze_headlines(headlines: List[str], batch_size: int = 16) -> List[Dict]:
    if not headlines:
        return []

    pipe = get_finbert_pipeline()

    try:
        results = pipe(headlines, batch_size=batch_size)
    except Exception as e:
        st.error(f"❌ FinBERT inference failed: {e}")
        raise

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

        influence_score = numeric

        out.append({
            "label": label,
            "score": score,
            "numeric_sentiment": numeric,
            "influence_score": influence_score,
            "raw_label": raw_label
        })

    return out
