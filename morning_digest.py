#!/usr/bin/env python3
"""
朝のテックダイジェスト
- HuggingFace Daily Papers
- GitHub Trending Python
- AI Blog (Google / DeepMind / Meta)
- Towards Data Science (Medium)
毎日 09:00 JST (= 00:00 UTC) に GitHub Actions で実行
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from time import mktime

import feedparser
import requests
from bs4 import BeautifulSoup
from google import genai

JST = timezone(timedelta(hours=9))
MODEL = "gemini-3.8-flash"

RSS_FEEDS = {
    "Google AI Blog":       "https://blog.google/technology/ai/rss/",
    "DeepMind Blog":        "https://deepmind.google/blog/rss.xml",
    "Meta AI Blog":         "https://engineering.fb.com/category/ai-research/feed/",
    "Towards Data Science": "https://medium.com/feed/towards-data-science",
}


# ── Fetch ─────────────────────────────────────────────────────────────────────

def fetch_hf_papers() -> list[dict]:
    print("Fetching HuggingFace Daily Papers...")
    resp = requests.get("https://huggingface.co/api/daily_papers", timeout=15)
    resp.raise_for_status()
    result = []
    for p in resp.json()[:6]:
        paper = p.get("paper", {})
        result.append({
            "title":    paper.get("title", ""),
            "abstract": paper.get("summary", "")[:400],
            "url":      f"https://huggingface.co/papers/{paper.get('id', '')}",
        })
    print(f"  {len(result)} papers found.")
    return result


def fetch_github_trending() -> list[dict]:
    print("Fetching GitHub Trending Python...")
    resp = requests.get(
        "https://github.com/trending/python?since=daily",
        headers={"Accept": "text/html", "User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    repos = []
    for article in soup.select("article.Box-row")[:5]:
        name_el = article.select_one("h2 a")
        if not name_el:
            continue
        path = name_el.get("href", "").strip("/")
        desc_el = article.select_one("p")
        desc = desc_el.get_text(strip=True) if desc_el else "No description"
        repos.append({
            "name": path.replace("/", " / "),
            "desc": desc,
            "url":  f"https://github.com/{path}",
        })
    print(f"  {len(repos)} repos found.")
    return repos


def fetch_rss(name: str, url: str, max_items: int = 3, max_age_hours: int = 48) -> list[dict]:
    try:
        feed = feedparser.parse(url)
        if feed.bozo and not feed.entries:
            print(f"  Warning: {name} feed parse error.")
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        items = []
        for entry in feed.entries:
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published = datetime.fromtimestamp(mktime(entry.published_parsed), tz=timezone.utc)
            if published and published < cutoff:
                continue
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:400]
            items.append({
                "title":   entry.get("title", ""),
                "summary": summary,
                "url":     entry.get("link", ""),
                "source":  name,
            })
            if len(items) >= max_items:
                break
        print(f"  {name}: {len(items)} items.")
        return items
    except Exception as e:
        print(f"  Warning: {name} failed: {e}")
        return []


# ── Summarize ─────────────────────────────────────────────────────────────────

def summarize(client, section: str, items: list[dict]) -> str:
    if not items:
        return ""
    text = "\n".join(
        f"- {item.get('title', '')}: {item.get('abstract') or item.get('summary') or item.get('desc', '')}"
        for item in items
    )
    prompt = (
        f"以下は「{section}」の最新情報です。\n"
        "データサイエンティスト向けに、各項目を1〜2文の日本語で簡潔に要約してください。\n"
        "何が新しいのか・なぜ重要かを含め、箇条書きで「• 」始めにしてください。\n\n"
        + text
    )
    for attempt in range(4):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            if attempt == 3:
                print(f"  Warning: summarize failed for '{section}': {e}")
                return ""
            wait = 15 * (attempt + 1)
            print(f"  Gemini busy, retrying in {wait}s... ({attempt + 1}/3)")
            time.sleep(wait)


# ── Slack ─────────────────────────────────────────────────────────────────────

def build_blocks(date_str: str, sections: list[dict]) -> list[dict]:
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🌅 朝のテックダイジェスト — {date_str}"},
        },
        {"type": "divider"},
    ]
    for sec in sections:
        if not sec.get("summary"):
            continue
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{sec['emoji']} {sec['title']}*\n{sec['summary']}",
            },
        })
        if sec.get("links"):
            link_text = "  |  ".join(
                f"<{url}|{label[:30]}>" for label, url in sec["links"] if url
            )
            blocks.append({
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": link_text}],
            })
        blocks.append({"type": "divider"})
    return blocks


def post_to_slack(webhook_url: str, blocks: list[dict]) -> None:
    resp = requests.post(webhook_url, json={"blocks": blocks}, timeout=15)
    resp.raise_for_status()
    print("Slack post OK.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not webhook:
        sys.exit("Error: SLACK_WEBHOOK_URL is not set.")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY is not set.")

    date_str = datetime.now(JST).strftime("%Y年%m月%d日")
    client = genai.Client(api_key=api_key)

    # 1. HuggingFace
    hf = fetch_hf_papers()
    hf_sum = summarize(client, "HuggingFace Daily Papers", hf)
    hf_links = [(p["title"][:30], p["url"]) for p in hf[:4]]

    # 2. GitHub Trending
    gh = fetch_github_trending()
    gh_sum = summarize(client, "GitHub Trending Python", gh)
    gh_links = [(r["name"], r["url"]) for r in gh[:4]]

    # 3. AI Blogs (Google / DeepMind / Meta)
    ai_items = []
    for name in ["Google AI Blog", "DeepMind Blog", "Meta AI Blog"]:
        ai_items.extend(fetch_rss(name, RSS_FEEDS[name]))
    blog_sum = summarize(client, "AI Blog (Google / DeepMind / Meta)", ai_items)
    blog_links = [(f"[{i['source'].split()[0]}] {i['title'][:25]}", i["url"]) for i in ai_items[:4]]

    # 4. Towards Data Science
    tds = fetch_rss("Towards Data Science", RSS_FEEDS["Towards Data Science"])
    tds_sum = summarize(client, "Towards Data Science", tds)
    tds_links = [(i["title"][:30], i["url"]) for i in tds[:4]]

    sections = [
        {"emoji": "📄", "title": "HuggingFace Daily Papers", "summary": hf_sum,   "links": hf_links},
        {"emoji": "⭐", "title": "GitHub Trending Python",   "summary": gh_sum,   "links": gh_links},
        {"emoji": "🤖", "title": "AI Blog (Google / DeepMind / Meta)", "summary": blog_sum, "links": blog_links},
        {"emoji": "📝", "title": "Towards Data Science",     "summary": tds_sum,  "links": tds_links},
    ]

    blocks = build_blocks(date_str, sections)
    print("Posting to Slack...")
    post_to_slack(webhook, blocks)


if __name__ == "__main__":
    main()
