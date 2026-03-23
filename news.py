"""
AI News Pipeline - Aggregates AI news from multiple sources and sends to Telegram.
Designed to run as a GitHub Actions cron job (free, no server needed).

Sources:
 1. ArXiv (cs.AI, cs.LG, cs.CL)  - Latest AI papers
 2. HuggingFace Blog RSS           - Model releases, tools
 3. Hacker News                    - Top AI/ML stories
 4. Reddit (via RSS)               - Community discussions
 5. OpenAI Blog RSS                - Product announcements
 6. Google AI Blog RSS             - Research updates
 7. Anthropic Blog RSS             - Claude / safety research
 8. MIT Tech Review AI RSS         - Industry news
 9. The Verge AI RSS               - Consumer AI news
10. VentureBeat AI RSS             - Enterprise AI
11. Papers With Code               - Trending ML papers
12. GitHub Trending (AI repos)     - Hot open-source tools

Notification: Telegram Bot
"""

import os
import json
import hashlib
import time
import re
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

# ─── Configuration ───────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN   = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.environ.get("TELEGRAM_CHAT_ID", "")
SEEN_CACHE_FILE      = os.environ.get("SEEN_CACHE_FILE", "seen_items.json")
MAX_ITEMS_PER_SOURCE = 5
MAX_MESSAGE_LENGTH   = 4000  # Telegram limit is 4096

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fetch_url(url, timeout=15, headers=None):
    """Fetch URL content with error handling."""
    try:
        h = {"User-Agent": "Mozilla/5.0 (compatible; AI-News-Bot/1.0)"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(url, headers=h)
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
    return text[:max_len] + "…" if len(text) > max_len else text

def parse_rss(raw, source, emoji, max_items=None):
    """Generic RSS/Atom parser. Returns list of item dicts."""
    if max_items is None:
        max_items = MAX_ITEMS_PER_SOURCE
    items = []
    if not raw:
        return items
    try:
        root = ET.fromstring(raw)

        # --- RSS 2.0 ---
        for item in root.findall(".//item")[:max_items]:
            title = item.find("title")
            link  = item.find("link")
            desc  = item.find("description")
            if title is not None and title.text:
                items.append({
                    "source":  source,
                    "emoji":   emoji,
                    "title":   title.text.strip(),
                    "summary": truncate(re.sub(r"<[^>]+>", "", (desc.text or "") if desc is not None else ""), 150),
                    "url":     (link.text or "").strip() if link is not None else "",
                })

        # --- Atom ---
        if not items:
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("a:entry", ns)[:max_items]:
                title   = entry.find("a:title", ns)
                link    = entry.find("a:link", ns)
                summary = entry.find("a:summary", ns)
                if title is not None and title.text:
                    items.append({
                        "source":  source,
                        "emoji":   emoji,
                        "title":   title.text.strip(),
                        "summary": truncate(re.sub(r"<[^>]+>", "", (summary.text or "") if summary is not None else ""), 150),
                        "url":     link.get("href", "") if link is not None else "",
                    })
    except ET.ParseError as e:
        print(f"  [WARN] XML parse error for {source}: {e}")
    return items

# ─── Source Fetchers ─────────────────────────────────────────────────────────

def fetch_arxiv():
    print("📄 Fetching ArXiv…")
    categories = ["cs.AI", "cs.LG", "cs.CL"]
    query = "+OR+".join([f"cat:{cat}" for cat in categories])
    url = (
        f"http://export.arxiv.org/api/query?search_query={query}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={MAX_ITEMS_PER_SOURCE}"
    )
    raw = fetch_url(url)
    items = []
    if not raw:
        print("  Found 0 papers"); return items
    try:
        root = ET.fromstring(raw)
        ns   = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title   = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            link    = entry.find("atom:id", ns)
            if title is not None and title.text:
                items.append({
                    "source":  "ArXiv",
                    "emoji":   "📄",
                    "title":   " ".join(title.text.strip().split()),
                    "summary": truncate(summary.text if summary is not None else "", 150),
                    "url":     link.text.strip() if link is not None else "",
                })
    except ET.ParseError as e:
        print(f"  [WARN] ArXiv parse error: {e}")
    print(f"  Found {len(items)} papers")
    return items


def fetch_huggingface():
    print("🤗 Fetching HuggingFace blog…")
    items = parse_rss(fetch_url("https://huggingface.co/blog/feed.xml"), "HuggingFace", "🤗")
    print(f"  Found {len(items)} posts")
    return items


def fetch_hackernews():
    """Use Algolia HN API — search multiple focused AI keywords."""
    print("🔶 Fetching Hacker News…")
    items = []
    seen_ids = set()

    queries = ["LLM", "GPT", "Claude AI", "Gemini AI", "open source AI", "AI agent"]
    for q in queries:
        url = (
            f"https://hn.algolia.com/api/v1/search_by_date?"
            f"query={urllib.parse.quote(q)}&tags=story"
            f"&hitsPerPage=5&numericFilters=points%3E5"
        )
        data = fetch_json(url)
        if not data:
            continue
        for hit in data.get("hits", []):
            oid = hit.get("objectID")
            if oid in seen_ids:
                continue
            seen_ids.add(oid)
            title = hit.get("title", "")
            if title:
                items.append({
                    "source":  "Hacker News",
                    "emoji":   "🔶",
                    "title":   title,
                    "summary": f"⬆️ {hit.get('points', 0)} points",
                    "url":     hit.get("url") or f"https://news.ycombinator.com/item?id={oid}",
                })
        if len(items) >= MAX_ITEMS_PER_SOURCE:
            break
        time.sleep(0.3)

    items = items[:MAX_ITEMS_PER_SOURCE]
    print(f"  Found {len(items)} stories")
    return items


def fetch_reddit():
    """Use Reddit RSS feeds — no API key, not blocked by GH Actions."""
    print("🔴 Fetching Reddit…")
    items = []
    subreddits = ["MachineLearning", "LocalLLaMA", "artificial", "singularity"]
    for sub in subreddits:
        raw = fetch_url(f"https://www.reddit.com/r/{sub}/hot/.rss?limit={MAX_ITEMS_PER_SOURCE}")
        parsed = parse_rss(raw, f"r/{sub}", "🔴")
        items.extend(parsed)
        time.sleep(0.5)
    print(f"  Found {len(items)} posts")
    return items


def fetch_openai_blog():
    print("🟢 Fetching OpenAI blog…")
    items = parse_rss(fetch_url("https://openai.com/blog/rss.xml"), "OpenAI", "🟢")
    print(f"  Found {len(items)} posts")
    return items


def fetch_google_ai():
    print("🔵 Fetching Google AI blog…")
    items = parse_rss(fetch_url("https://blog.google/technology/ai/rss/"), "Google AI", "🔵")
    print(f"  Found {len(items)} posts")
    return items


def fetch_anthropic():
    print("🟣 Fetching Anthropic blog…")
    items = parse_rss(fetch_url("https://www.anthropic.com/rss.xml"), "Anthropic", "🟣")
    print(f"  Found {len(items)} posts")
    return items


def fetch_mit_tech_review():
    print("📰 Fetching MIT Tech Review AI…")
    items = parse_rss(fetch_url("https://www.technologyreview.com/feed/"), "MIT Tech Review", "📰")
    # Filter to only AI-related items
    ai_keywords = re.compile(r"\b(AI|LLM|GPT|model|neural|machine learning|deep learning|chatbot|robot|autonomous)\b", re.I)
    items = [i for i in items if ai_keywords.search(i["title"] + i["summary"])][:MAX_ITEMS_PER_SOURCE]
    print(f"  Found {len(items)} posts")
    return items


def fetch_verge_ai():
    print("⚡ Fetching The Verge AI…")
    items = parse_rss(fetch_url("https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"), "The Verge AI", "⚡")
    print(f"  Found {len(items)} posts")
    return items


def fetch_venturebeat_ai():
    print("💼 Fetching VentureBeat AI…")
    items = parse_rss(fetch_url("https://venturebeat.com/category/ai/feed/"), "VentureBeat AI", "💼")
    print(f"  Found {len(items)} posts")
    return items


def fetch_papers_with_code():
    """Fetch trending ML papers from Papers With Code."""
    print("📊 Fetching Papers With Code…")
    items = []
    data  = fetch_json("https://paperswithcode.com/api/v1/papers/?ordering=-published&items_per_page=5")
    if not data:
        print("  Found 0 papers"); return items
    for paper in data.get("results", [])[:MAX_ITEMS_PER_SOURCE]:
        title = paper.get("title", "")
        url   = paper.get("url_pdf") or paper.get("paper_url", "")
        if title:
            items.append({
                "source":  "Papers w/ Code",
                "emoji":   "📊",
                "title":   title,
                "summary": truncate(paper.get("abstract", ""), 150),
                "url":     url,
            })
    print(f"  Found {len(items)} papers")
    return items


def fetch_github_trending():
    """Scrape GitHub trending page for AI/ML repos."""
    print("🐙 Fetching GitHub Trending AI…")
    items = []
    raw   = fetch_url("https://github.com/trending?l=python&since=daily")
    if not raw:
        print("  Found 0 repos"); return items

    ai_keywords = re.compile(r"\b(AI|LLM|GPT|RAG|agent|diffusion|neural|transformer|vision|NLP|chatbot|embedding)\b", re.I)
    # Extract repo names and descriptions from HTML
    desc_pattern = re.compile(r'<p[^>]*class="[^"]*col-9[^"]*"[^>]*>\s*([^<]+)\s*</p>')

    repos = re.findall(r'<h2[^>]*>\s*<a[^>]+href="/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)"', raw)
    descs = desc_pattern.findall(raw)

    for i, repo in enumerate(repos[:20]):
        desc = descs[i].strip() if i < len(descs) else ""
        if ai_keywords.search(repo + " " + desc):
            items.append({
                "source":  "GitHub Trending",
                "emoji":   "🐙",
                "title":   repo,
                "summary": truncate(desc, 150),
                "url":     f"https://github.com/{repo}",
            })
        if len(items) >= MAX_ITEMS_PER_SOURCE:
            break

    print(f"  Found {len(items)} repos")
    return items


# ─── Message Formatting ──────────────────────────────────────────────────────

def format_digest(new_items):
    """Format items into Telegram HTML digest. Splits into multiple messages if needed."""
    if not new_items:
        return []

    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"<b>🤖 AI News Digest</b>\n<i>{now}</i>\n"]

    by_source = {}
    for item in new_items:
        by_source.setdefault(item["source"], []).append(item)

    for source, source_items in by_source.items():
        emoji = source_items[0]["emoji"]
        lines.append(f"\n<b>{emoji} {source}</b>")
        for item in source_items:
            if item["url"]:
                lines.append(f'  • <a href="{item["url"]}">{item["title"]}</a>')
            else:
                lines.append(f"  • {item['title']}")
            if item["summary"]:
                lines.append(f"    <i>{item['summary']}</i>")

    lines.append("\n—\n<i>🔄 Next update in 30 min</i>")
    full = "\n".join(lines)

    # Split into chunks if message is too long
    messages = []
    while len(full) > MAX_MESSAGE_LENGTH:
        split_at = full.rfind("\n", 0, MAX_MESSAGE_LENGTH)
        if split_at == -1:
            split_at = MAX_MESSAGE_LENGTH
        messages.append(full[:split_at])
        full = "<i>(continued…)</i>\n" + full[split_at:]
    messages.append(full)
    return messages


# ─── Telegram Sender ─────────────────────────────────────────────────────────

def send_telegram(message):
    """Send a single message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Secrets not set — printing preview:")
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
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read().decode())
            if result.get("ok"):
                print("✅ Message sent to Telegram!")
                return True
            print(f"❌ Telegram API error: {result}")
            return False
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
        return False


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print(f"🚀 AI News Pipeline - {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    seen = load_seen()
    print(f"📦 Loaded {len(seen)} previously seen items\n")

    fetchers = [
        fetch_arxiv,
        fetch_huggingface,
        fetch_hackernews,
        fetch_reddit,
        fetch_openai_blog,
        fetch_google_ai,
        fetch_anthropic,
        fetch_mit_tech_review,
        fetch_verge_ai,
        fetch_venturebeat_ai,
        fetch_papers_with_code,
        fetch_github_trending,
    ]

    all_items = []
    for fetcher in fetchers:
        try:
            all_items.extend(fetcher())
        except Exception as e:
            print(f"  [ERROR] {fetcher.__name__}: {e}")
        time.sleep(0.5)

    print(f"\n📊 Total fetched: {len(all_items)}")

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
        messages = format_digest(new_items)
        for msg in messages:
            send_telegram(msg)
            time.sleep(1)  # avoid Telegram rate limit
    else:
        print("ℹ️  No new items — skipping notification")

    print("\n✅ Pipeline complete!")
    return len(new_items)


if __name__ == "__main__":
    run_pipeline()
