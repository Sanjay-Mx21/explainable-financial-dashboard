# utils/news_loader.py
import feedparser
from datetime import datetime, timezone
import time

def parse_rss_feed(url: str, max_items: int = 50):
    feed = feedparser.parse(url)
    items = []

    for e in feed.entries[:max_items]:
        published = None
        if hasattr(e, "published_parsed") and e.published_parsed:
            published = datetime.fromtimestamp(
                time.mktime(e.published_parsed), tz=timezone.utc
            )
        elif hasattr(e, "updated_parsed") and e.updated_parsed:
            published = datetime.fromtimestamp(
                time.mktime(e.updated_parsed), tz=timezone.utc
            )

        title = getattr(e, "title", "") or ""
        summary = getattr(e, "summary", "") or ""

        content = summary.strip()
        if len(content) < 20:
            content = title.strip()

        items.append({
            "published": published,
            "title": title,
            "summary": summary,
            "content": content,
            "link": getattr(e, "link", "") or "",
            "source": feed.feed.get("title", "")
        })

    return items
