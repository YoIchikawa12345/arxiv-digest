#!/usr/bin/env python3
"""
朝のテックダイジェスト
- HuggingFace Daily Papers
- GitHub Trending Python
- AI Blog (Google / DeepMind / Meta)
- Towards Data Science (Medium)
毎日 09:00 JST (= 00:00 UTC) に GitHub Actions で実行
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from time import mktime

import feedparser
import requests
from bs4 import BeautifulSoup
from google import genai

JST = timezone(timedelta(hours=9))
GEMINI_MODEL = "gemini-3.8-flash"

RSS_FEEDS = {
    "Google AI Blog":       "https://blog.google/technology/ai/rss/",
    "DeepMind Blog":        "https://deepmind.google/blog/rss.xml",
    "Meta AI Blog":         "https://engineering.fb.com/category/ai-research/feed/",
    "Towards Data Science": "https://medium.com/feed/towards-data-science",
}

NHK_FEEDS = {
    "社会":   "https://www.nhk.or.jp/rss/news/cat1.xml",
    "国際":   "https://www.nhk.or.jp/rss/news/cat6.xml",
    "経済":   "https://www.nhk.or.jp/rss/news/cat5.xml",
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

def call_gemini(api_key: str, prompt: str) -> str:
    """google-genai SDK で Gemini を呼び出す。エラー時は最大2回リトライ。"""
    client = genai.Client(api_key=api_key)
    for attempt in range(3):
        try:
            message = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            return message.text.strip()
        except Exception as e:
            if attempt == 2:
                print(f"  Warning: Gemini error: {e}")
                return ""
            wait = 30 * (attempt + 1)
            print(f"  Error, retrying in {wait}s... ({attempt + 1}/2)")
            time.sleep(wait)
    return ""


def summarize_all(api_key: str, sections: dict[str, list[dict]]) -> dict[str, str]:
    """全セクションを1回のAPI呼び出しでまとめて要約する。JSON形式で返させる。"""
    keys = list(sections.keys())
    parts = []
    for i, (section, items) in enumerate(sections.items()):
        if not items:
            continue
        text = "\n".join(
            f"- {item.get('title', '')}: {item.get('abstract') or item.get('summary') or item.get('desc', '')}"
            for item in items
        )
        parts.append(f"[{i}] {section}\n{text}")

    result = {k: "" for k in keys}
    if not parts:
        return result

    prompt = (
        "以下の複数カテゴリの最新情報を、AIに詳しくないバックエンドエンジニア向けに日本語で解説してください。\n"
        "各カテゴリごとに、各項目を「• タイトル\\n　解説」の形式で箇条書きにしてください。\n\n"
        "解説のガイドライン:\n"
        "- 2〜3文で「どんな問題を・どうやって解決・なぜ重要か（数値があれば）」を書く\n"
        "- 専門用語は使う場合かっこ内で簡単に補足する（例：LLM（大規模言語モデル））\n"
        "- 初めてその分野に触れる人でもイメージできる平易な言葉で書く\n"
        "- 具体的な数値や比較があれば必ず含める\n\n"
        "必ず以下のJSON形式のみで回答し、他のテキストは含めないでください:\n"
        '{"0": "• Title\\n　解説\\n• Title\\n　解説", "1": "...", ...}\n\n'
        "カテゴリ番号と内容:\n"
        + "\n\n".join(parts)
    )
    print("Calling Gemini API (single request for all sections)...")
    raw = call_gemini(api_key, prompt)
    print(f"  Raw response length: {len(raw)} chars")

    if not raw:
        return result

    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        print(f"  Warning: no JSON found in response. Raw:\n{raw[:300]}")
        return result
    try:
        data = json.loads(json_match.group())
        for i, key in enumerate(keys):
            result[key] = data.get(str(i), "")
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse error: {e}. Raw:\n{raw[:300]}")
    return result


# ── Slack ─────────────────────────────────────────────────────────────────────

def build_raw_summary(items: list[dict]) -> str:
    """Gemini失敗時のフォールバック: タイトル一覧を箇条書きにする。"""
    lines = []
    for item in items[:5]:
        title = item.get("title", item.get("name", ""))
        desc = item.get("abstract", item.get("summary", item.get("desc", "")))
        if desc:
            desc = (desc[:80] + "…") if len(desc) > 80 else desc
            lines.append(f"• {title}: {desc}")
        else:
            lines.append(f"• {title}")
    return "\n".join(lines)


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

    # データ収集
    hf = fetch_hf_papers()
    hf_links = [(p["title"][:30], p["url"]) for p in hf[:4]]

    gh = fetch_github_trending()
    gh_links = [(r["name"], r["url"]) for r in gh[:4]]

    ai_items = []
    for name in ["Google AI Blog", "DeepMind Blog", "Meta AI Blog"]:
        ai_items.extend(fetch_rss(name, RSS_FEEDS[name]))
    blog_links = [(f"[{i['source'].split()[0]}] {i['title'][:25]}", i["url"]) for i in ai_items[:4]]

    tds = fetch_rss("Towards Data Science", RSS_FEEDS["Towards Data Science"])
    tds_links = [(i["title"][:30], i["url"]) for i in tds[:4]]

    nhk_items = []
    for cat, url in NHK_FEEDS.items():
        items = fetch_rss(f"NHK {cat}", url, max_items=2, max_age_hours=24)
        nhk_items.extend(items)
    nhk_links = [(i["title"][:30], i["url"]) for i in nhk_items[:4]]

    # Gemini API を1回だけ呼ぶ
    sums = summarize_all(api_key, {
        "HuggingFace Daily Papers":        hf,
        "GitHub Trending Python":          gh,
        "AI Blog (Google / DeepMind / Meta)": ai_items,
        "Towards Data Science":            tds,
        "NHK ニュース（国内・国際・経済）":      nhk_items,
    })

    sections = [
        {"emoji": "📄", "title": "HuggingFace Daily Papers",
         "summary": sums["HuggingFace Daily Papers"] or build_raw_summary(hf),
         "links": hf_links},
        {"emoji": "⭐", "title": "GitHub Trending Python",
         "summary": sums["GitHub Trending Python"] or build_raw_summary(gh),
         "links": gh_links},
        {"emoji": "🤖", "title": "AI Blog (Google / DeepMind / Meta)",
         "summary": sums["AI Blog (Google / DeepMind / Meta)"] or build_raw_summary(ai_items),
         "links": blog_links},
        {"emoji": "📝", "title": "Towards Data Science",
         "summary": sums["Towards Data Science"] or build_raw_summary(tds),
         "links": tds_links},
        {"emoji": "📰", "title": "NHK ニュース（国内・国際・経済）",
         "summary": sums["NHK ニュース（国内・国際・経済）"] or build_raw_summary(nhk_items),
         "links": nhk_links},
    ]

    blocks = build_blocks(date_str, sections)
    print("Posting to Slack...")
    post_to_slack(webhook, blocks)


if __name__ == "__main__":
    main()
