"""
AI News Pipeline - Aggregates AI news from multiple sources and sends to Telegram.
Designed to run as a GitHub Actions cron job (free, no server needed).

Sources:
1. ArXiv (cs.AI, cs.LG, cs.CL)  - Latest AI papers
2. HuggingFace Blog RSS           - Model releases, tools
3. Hacker News                    - Top AI/ML stories
4. Reddit r/MachineLearning etc.  - Community discussions
5. OpenAI Blog RSS                - Product announcements
6. Google AI Blog RSS             - Research updates

Notification: Telegram Bot
"""

import os
import json
import hashlib
import time
import re
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

# ─── Configuration ───────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
SEEN_CACHE_FILE    = os.environ.get("SEEN_CACHE_FILE", "seen_items.json")
MAX_ITEMS_PER_SOURCE = 5
MAX_MESSAGE_LENGTH   = 4000  # Telegram limit is 4096

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fetch_url(url, timeout=15):
    """Fetch URL content with error handling."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "AI-News-Bot/1.0 (github-actions)"
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"  [WARN] Failed to fetch {url}: {e}")
        return None

def fetch_json(url, timeout=15):
    """Fetch and parse JSON from URL."""
    raw = fetch_url(url, timeout)
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON decode error for {url}: {e}")
    return None

def item_hash(text):
    """Create a short hash for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:12]

def load_seen():
    """Load previously seen item hashes."""
    if os.path.exists(SEEN_CACHE_FILE):
        try:
            with open(SEEN_CACHE_FILE, "r") as f:
                data = json.load(f)
            # Keep only last 7 days of hashes
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            return {k: v for k, v in data.items() if v > cutoff}
        except Exception:
            pass
    return {}

def save_seen(seen):
    """Save seen item hashes to file."""
    with open(SEEN_CACHE_FILE, "w") as f:
        json.dump(seen, f)

def truncate(text, max_len=200):
    """Truncate text to max length."""
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text

# ─── Source Fetchers ─────────────────────────────────────────────────────────

def fetch_arxiv():
    """Fetch latest AI papers from ArXiv."""
    print("📄 Fetching ArXiv papers…")
    items = []

    categories = ["cs.AI", "cs.LG", "cs.CL"]
    query = "+OR+".join([f"cat:{cat}" for cat in categories])
    url = (
        f"http://export.arxiv.org/api/query?search_query={query}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={MAX_ITEMS_PER_SOURCE}"
    )

    raw = fetch_url(url)
    if not raw:
        return items

    try:
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title   = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            link    = entry.find("atom:id", ns)

            if title is not None:
                items.append({
                    "source":  "ArXiv",
                    "emoji":   "📄",
                    "title":   " ".join(title.text.strip().split()),
                    "summary": truncate(summary.text if summary is not None else "", 150),
                    "url":     link.text.strip() if link is not None else "",
                })
    except ET.ParseError as e:
        print(f"  [WARN] ArXiv XML parse error: {e}")

    print(f"  Found {len(items)} papers")
    return items


def fetch_huggingface():
    """Fetch latest from HuggingFace blog RSS."""
    print("🤗 Fetching HuggingFace blog…")
    items = []
    url = "https://huggingface.co/blog/feed.xml"

    raw = fetch_url(url)
    if not raw:
        return items

    try:
        root = ET.fromstring(raw)
        # Try RSS format first
        for item in root.findall(".//item")[:MAX_ITEMS_PER_SOURCE]:
            title = item.find("title")
            link  = item.find("link")
            desc  = item.find("description")

            if title is not None:
                items.append({
                    "source":  "HuggingFace",
                    "emoji":   "🤗",
                    "title":   title.text.strip() if title.text else "Untitled",
                    "summary": truncate(re.sub(r"<[^>]+>", "", desc.text or ""), 150),
                    "url":     link.text.strip() if link is not None and link.text else "",
                })

        # Fallback: Atom format
        if not items:
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns)[:MAX_ITEMS_PER_SOURCE]:
                title   = entry.find("atom:title", ns)
                link    = entry.find("atom:link", ns)
                summary = entry.find("atom:summary", ns)

                if title is not None:
                    items.append({
                        "source":  "HuggingFace",
                        "emoji":   "🤗",
                        "title":   title.text.strip() if title.text else "Untitled",
                        "summary": truncate(re.sub(r"<[^>]+>", "", (summary.text or "")), 150),
                        "url":     link.get("href", "") if link is not None else "",
                    })
    except ET.ParseError as e:
        print(f"  [WARN] HuggingFace XML parse error: {e}")

    print(f"  Found {len(items)} posts")
    return items


def fetch_hackernews():
    """Fetch top AI/ML stories from Hacker News."""
    print("🔶 Fetching Hacker News AI stories…")
    items = []

    keywords = "artificial intelligence OR LLM OR GPT OR machine learning OR neural network OR AI model"
    encoded  = urllib.parse.quote(keywords)
    url = (
        f"https://hn.algolia.com/api/v1/search_by_date?"
        f"query={encoded}&tags=story&hitsPerPage={MAX_ITEMS_PER_SOURCE * 2}"
        f"&numericFilters=points>10"
    )

    data = fetch_json(url)
    if not data or "hits" not in data:
        return items

    for hit in data["hits"][:MAX_ITEMS_PER_SOURCE]:
        title     = hit.get("title", "")
        story_url = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
        points    = hit.get("points", 0)

        if title:
            items.append({
                "source":  "Hacker News",
                "emoji":   "🔶",
                "title":   title,
                "summary": f"⬆️ {points} points",
                "url":     story_url,
            })

    print(f"  Found {len(items)} stories")
    return items


def fetch_reddit():
    """Fetch hot posts from r/MachineLearning, r/artificial, r/LocalLLaMA."""
    print("🔴 Fetching Reddit AI communities…")
    items = []

    subreddits = ["MachineLearning", "artificial", "LocalLLaMA"]

    for sub in subreddits:
        url  = f"https://www.reddit.com/r/{sub}/hot.json?limit={MAX_ITEMS_PER_SOURCE}"
        data = fetch_json(url)

        if not data or "data" not in data:
            continue

        for post in data["data"].get("children", [])[:MAX_ITEMS_PER_SOURCE]:
            pdata     = post.get("data", {})
            title     = pdata.get("title", "")
            permalink = pdata.get("permalink", "")
            ups       = pdata.get("ups", 0)
            flair     = pdata.get("link_flair_text", "")

            if title and not pdata.get("stickied", False):
                flair_text = f"[{flair}] " if flair else ""
                items.append({
                    "source":  f"r/{sub}",
                    "emoji":   "🔴",
                    "title":   f"{flair_text}{title}",
                    "summary": f"⬆️ {ups} upvotes",
                    "url":     f"https://reddit.com{permalink}" if permalink else "",
                })

        time.sleep(1)  # Be nice to Reddit API

    print(f"  Found {len(items)} posts")
    return items


def fetch_openai_blog():
    """Fetch latest from OpenAI blog RSS."""
    print("🟢 Fetching OpenAI blog…")
    items = []
    url   = "https://openai.com/blog/rss.xml"

    raw = fetch_url(url)
    if not raw:
        return items

    try:
        root = ET.fromstring(raw)
        for item in root.findall(".//item")[:MAX_ITEMS_PER_SOURCE]:
            title = item.find("title")
            link  = item.find("link")
            desc  = item.find("description")

            if title is not None and title.text:
                items.append({
                    "source":  "OpenAI",
                    "emoji":   "🟢",
                    "title":   title.text.strip(),
                    "summary": truncate(re.sub(r"<[^>]+>", "", desc.text or ""), 150),
                    "url":     link.text.strip() if link is not None and link.text else "",
                })
    except ET.ParseError:
        print("  [WARN] OpenAI RSS parse error")

    print(f"  Found {len(items)} posts")
    return items


def fetch_google_ai():
    """Fetch latest from Google AI blog."""
    print("🔵 Fetching Google AI blog…")
    items = []
    url   = "https://blog.google/technology/ai/rss/"

    raw = fetch_url(url)
    if not raw:
        return items

    try:
        root = ET.fromstring(raw)
        for item in root.findall(".//item")[:MAX_ITEMS_PER_SOURCE]:
            title = item.find("title")
            link  = item.find("link")
            desc  = item.find("description")

            if title is not None and title.text:
                items.append({
                    "source":  "Google AI",
                    "emoji":   "🔵",
                    "title":   title.text.strip(),
                    "summary": truncate(re.sub(r"<[^>]+>", "", desc.text or ""), 150),
                    "url":     link.text.strip() if link is not None and link.text else "",
                })
    except ET.ParseError:
        print("  [WARN] Google AI RSS parse error")

    print(f"  Found {len(items)} posts")
    return items


# ─── Message Formatting ──────────────────────────────────────────────────────

def format_digest(new_items):
    """Format items into a Telegram-friendly digest message (HTML)."""
    if not new_items:
        return None

    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"<b>🤖 AI News Digest</b>\n<i>{now}</i>\n"]

    # Group by source
    by_source = {}
    for item in new_items:
        src = item["source"]
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(item)

    for source, source_items in by_source.items():
        emoji = source_items[0]["emoji"]
        lines.append(f"\n<b>{emoji} {source}</b>")

        for item in source_items:
            title   = item["title"]
            url     = item["url"]
            summary = item["summary"]

            if url:
                lines.append(f'  • <a href="{url}">{title}</a>')
            else:
                lines.append(f"  • {title}")

            if summary:
                lines.append(f"    <i>{summary}</i>")

    lines.append("\n—\n<i>🔄 Next update in 30 min</i>")

    message = "\n".join(lines)

    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH - 20] + "\n\n<i>... truncated</i>"

    return message


# ─── Telegram Sender ─────────────────────────────────────────────────────────

def send_telegram(message):
    """Send message to Telegram using Bot API (no dependencies needed)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set!")
        print("   Message preview:")
        print(message[:500])
        return False

    url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({
        "chat_id":                  TELEGRAM_CHAT_ID,
        "text":                     message,
        "parse_mode":               "HTML",
        "disable_web_page_preview": True,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                print("✅ Message sent to Telegram!")
                return True
            else:
                print(f"❌ Telegram API error: {result}")
                return False
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
        return False


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    """Main pipeline: fetch all sources, deduplicate, send digest."""
    print("=" * 60)
    print(f"🚀 AI News Pipeline - {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    seen = load_seen()
    print(f"📦 Loaded {len(seen)} previously seen items\n")

    all_items = []
    fetchers  = [
        fetch_arxiv,
        fetch_huggingface,
        fetch_hackernews,
        fetch_reddit,
        fetch_openai_blog,
        fetch_google_ai,
    ]

    for fetcher in fetchers:
        try:
            items = fetcher()
            all_items.extend(items)
        except Exception as e:
            print(f"  [ERROR] {fetcher.__name__}: {e}")
        time.sleep(0.5)

    print(f"\n📊 Total items fetched: {len(all_items)}")

    # Deduplicate
    new_items = []
    now_iso   = datetime.now(timezone.utc).isoformat()

    for item in all_items:
        h = item_hash(item["title"] + item.get("url", ""))
        if h not in seen:
            new_items.append(item)
            seen[h] = now_iso

    print(f"🆕 New items: {len(new_items)}")

    save_seen(seen)

    if new_items:
        message = format_digest(new_items)
        if message:
            send_telegram(message)
    else:
        print("ℹ️  No new items — skipping Telegram notification")

    print("\n✅ Pipeline complete!")
    return len(new_items)


if __name__ == "__main__":
    run_pipeline()
