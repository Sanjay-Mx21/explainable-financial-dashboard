# test_sentiment.py
from utils.sentiment_analysis import analyze_headlines

heads = [
    "Apple reports record revenue in Q3, shares jump",
    "Company X misses earnings by a wide margin, stock plunges",
    "Market opens flat amid mixed signals"
]

res = analyze_headlines(heads, batch_size=3)
for h, r in zip(heads, res):
    print("HEADLINE:", h)
    print("RESULT:", r)
    print("---")
