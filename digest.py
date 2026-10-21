#!/usr/bin/env python3
"""
ArXiv daily digest — GitHub Pages + Slack

Flow:
  1. Fetch today's NewsClaw paper list
  2. Fetch abstracts from arXiv API
  3. Ask Claude to generate: short summary, overview, key points, term glossary
  4. Render a standalone HTML page and write it to docs/
  5. Post a brief digest to Slack with a link to the HTML page

Required env vars:
  SLACK_WEBHOOK_URL    Slack Incoming Webhook URL
  ANTHROPIC_API_KEY    Anthropic API key
  GITHUB_REPOSITORY    Set automatically in GitHub Actions (owner/repo)
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google import genai
import requests
from bs4 import BeautifulSoup

JST = timezone(timedelta(hours=9))
NEWSCLAW_BASE = "https://canon-so8.github.io/news-claw/arxiv"
SLUG = "11-59-neta-trend-hf"
ARXIV_API = "https://export.arxiv.org/api/query"
MODEL = "gemini-3.8-flash"

TAG_COLORS = {
    "AGENT": "#3b82f6", "LLM": "#8b5cf6", "CV": "#10b981",
    "NLP": "#f97316",   "ML": "#06b6d4",  "AUDIO": "#ec4899",
    "QUANT": "#84cc16", "OTHER": "#6b7280",
}


# ── Fetch ─────────────────────────────────────────────────────────────────────

def today_url() -> tuple[str, str]:
    now = datetime.now(JST)
    date_str = now.strftime("%Y-%m-%d")
    url = f"{NEWSCLAW_BASE}/{now.year}/{now.month:02d}/{now.day:02d}/{SLUG}/"
    return url, date_str


def fetch_papers(page_url: str) -> list[dict]:
    resp = requests.get(page_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    papers = []
    article = soup.find("article")
    if not article:
        return papers

    for link in article.find_all("a", href=re.compile(r"arxiv\.org/abs/")):
        title = link.get_text(strip=True)
        if not title:
            continue
        m = re.search(r"abs/([^v/\s]+)", link["href"])
        if not m:
            continue
        arxiv_id = m.group(1)

        tags, votes, github_url = [], 0, None
        sib = link.find_next_sibling()
        if sib:
            for el in sib.find_all(True):
                txt = el.get_text(strip=True)
                if re.fullmatch(r"AGENT|LLM|CV|NLP|ML|AUDIO|QUANT|OTHER", txt, re.I):
                    upper = txt.upper()
                    if upper not in tags:
                        tags.append(upper)
                elif vm := re.match(r"▲\s*(\d+)", txt):
                    votes = int(vm.group(1))
            gh = sib.find("a", href=re.compile(r"github\.com"))
            if gh:
                github_url = gh["href"]

        papers.append({
            "title": title,
            "arxiv_id": arxiv_id,
            "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
            "tags": tags,
            "votes": votes,
            "github_url": github_url,
        })

    return papers[:10]


def fetch_abstracts(papers: list[dict]) -> dict[str, str]:
    ids = [p["arxiv_id"] for p in papers]
    resp = requests.get(
        ARXIV_API,
        params={"id_list": ",".join(ids), "max_results": len(ids)},
        timeout=30,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.text)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    result = {}
    for entry in root.findall("a:entry", ns):
        eid = entry.find("a:id", ns).text
        m = re.search(r"abs/([^v/\s]+)", eid)
        if m:
            summary_el = entry.find("a:summary", ns)
            if summary_el is not None:
                result[m.group(1)] = " ".join(summary_el.text.split())
    return result


# ── Generate content via Claude API ──────────────────────────────────────────

def generate_content(papers: list[dict], abstracts: dict[str, str]) -> list[dict]:
    """
    Returns list of dicts:
      id       : 1-based index
      short    : 1-line summary for Slack (≤40 chars)
      overview : 3-4 sentence beginner-friendly overview
      points   : list[str] — 3-4 key takeaways with concrete numbers
      terms    : list[{word, desc}] — glossary of 3-5 key terms
    """
    items = []
    for i, p in enumerate(papers, 1):
        abstract = abstracts.get(p["arxiv_id"], "Abstract unavailable.")
        items.append(f"[{i}] {p['title']}\n{abstract[:1000]}")

    prompt = (
        "以下の各論文について、AIに詳しくないバックエンドエンジニア向けに解説を生成してください。\n\n"
        "各論文に対して、以下の構造のJSONを返してください：\n"
        "[\n"
        "  {\n"
        "    \"id\": 1,\n"
        "    \"short\": \"Slack用の1行要約（40文字以内、数値を入れる）\",\n"
        "    \"overview\": \"3〜4文の初心者向け概要。『何の問題を・どうやって・どんな結果』を平易な言葉で。専門用語は最小限に、使う場合はすぐ説明する。\",\n"
        "    \"points\": [\n"
        "      \"核心的ポイント（具体的な数値や手法を含む）\",\n"
        "      \"...（3〜4項目）\"\n"
        "    ],\n"
        "    \"terms\": [\n"
        "      {\"word\": \"技術用語\", \"desc\": \"1〜2文の平易な解説。例え話を使ってOK。\"}\n"
        "    ]\n"
        "  },\n"
        "  ...\n"
        "]\n\n"
        "ガイドライン：\n"
        "- overviewは「なぜそれが必要か」から始める\n"
        "- pointsは「何がすごいのか」を数値で示す（例: ○○が△倍速い）\n"
        "- termsはその論文を読む上で絶対に知っておくべき用語を3〜5語選ぶ\n"
        "- 全体として「読んだだけで雰囲気がつかめる」ことを目標に\n\n"
        + "\n\n".join(items)
    )

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    message = client.models.generate_content(model=MODEL, contents=prompt)

    text = message.text
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if not json_match:
        return []
    try:
        return json.loads(json_match.group())
    except json.JSONDecodeError:
        return []


# ── HTML rendering ────────────────────────────────────────────────────────────

CSS = """
:root {
  --bg: #0f1623; --bg-card: #1a2234; --bg-section: #1e2a40;
  --border: rgba(255,255,255,0.07); --fg: #e2e8f0;
  --fg-muted: #7a90b8; --fg-dim: #4a5b7a;
  --accent: #f59e0b; --accent-sub: rgba(245,158,11,0.1);
  --green: #10b981; --green-sub: rgba(16,185,129,0.08);
  --purple: #8b5cf6; --purple-sub: rgba(139,92,246,0.08);
  --font-d: 'Source Serif 4', Georgia, serif;
  --font-b: 'IBM Plex Sans', system-ui, sans-serif;
  --font-m: 'IBM Plex Mono', monospace;
  color-scheme: dark;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #f5f7fb; --bg-card: #fff; --bg-section: #f0f4ff;
    --border: rgba(0,0,0,0.08); --fg: #1a2035;
    --fg-muted: #556080; --fg-dim: #9aa5bf;
    --accent: #b45309; --accent-sub: rgba(180,83,9,0.08);
    --green: #047857; --green-sub: rgba(4,120,87,0.07);
    --purple: #6d28d9; --purple-sub: rgba(109,40,217,0.07);
    color-scheme: light;
  }
}
*, *::before, *::after { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--fg);
  font-family: var(--font-b); font-size: 15px; line-height: 1.7;
  margin: 0; padding: 0;
}
a { color: var(--accent); }
.site-header {
  position: sticky; top: 0; z-index: 100;
  background: var(--bg); border-bottom: 1px solid var(--border);
  padding: 14px max(16px, 2vw);
}
.header-inner {
  max-width: 860px; margin: 0 auto;
  display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
}
.header-eyebrow {
  font-family: var(--font-m); font-size: 0.65rem; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent);
}
.header-title {
  font-family: var(--font-d); font-size: 1.15rem; font-weight: 600; flex: 1;
}
.header-meta { font-family: var(--font-m); font-size: 0.72rem; color: var(--fg-muted); }
.header-link {
  font-family: var(--font-m); font-size: 0.7rem; color: var(--fg-muted);
  text-decoration: none; border: 1px solid var(--border); padding: 3px 8px; border-radius: 4px;
}
.header-link:hover { color: var(--accent); border-color: var(--accent); }
.main { max-width: 860px; margin: 0 auto; padding: 20px max(16px, 2vw) 64px; }
.section-label-top {
  font-family: var(--font-m); font-size: 0.7rem; color: var(--fg-dim);
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 16px;
}
.paper-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: 10px; margin-bottom: 16px; overflow: hidden;
}
.card-top {
  display: grid; grid-template-columns: 50px 1fr;
  gap: 0 14px; padding: 18px 18px 14px;
}
.vote-block {
  display: flex; flex-direction: column; align-items: center; padding-top: 4px; gap: 2px;
}
.vote-arrow { font-size: 0.6rem; color: var(--accent); }
.vote-num {
  font-family: var(--font-m); font-size: 0.95rem; color: var(--accent);
  font-variant-numeric: tabular-nums; line-height: 1.1;
}
.rank { font-family: var(--font-m); font-size: 0.6rem; color: var(--fg-dim); }
.paper-title {
  font-family: var(--font-d); font-size: 1rem; font-weight: 600;
  line-height: 1.4; margin: 0 0 8px; text-wrap: balance;
}
.paper-title a { color: var(--fg); text-decoration: none; }
.paper-title a:hover { color: var(--accent); }
.tags { display: flex; gap: 4px; flex-wrap: wrap; }
.tag {
  font-family: var(--font-m); font-size: 0.63rem; padding: 2px 7px;
  border-radius: 3px; color: #fff; letter-spacing: 0.03em;
}
.card-sections { padding: 0 18px 18px 82px; display: flex; flex-direction: column; gap: 14px; }
@media (max-width: 560px) {
  .card-top { grid-template-columns: 44px 1fr; }
  .card-sections { padding: 0 14px 14px; }
}
.section-block {
  background: var(--bg-section); border-radius: 6px;
  border-left: 3px solid var(--accent); padding: 12px 14px;
}
.section-block.points { border-left-color: var(--green); }
.section-block.terms  { border-left-color: var(--purple); }
.section-label {
  font-family: var(--font-m); font-size: 0.68rem; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--fg-dim); margin-bottom: 8px;
}
.section-block p { margin: 0; font-size: 0.9rem; }
.points-list {
  margin: 0; padding-left: 1.2em;
  display: flex; flex-direction: column; gap: 4px;
}
.points-list li { font-size: 0.88rem; }
.terms-grid { display: flex; flex-direction: column; gap: 10px; }
.term-item { display: grid; grid-template-columns: auto 1fr; gap: 0 10px; align-items: baseline; }
.term-word {
  font-family: var(--font-m); font-size: 0.78rem; font-weight: 600;
  color: var(--purple); white-space: nowrap; padding-top: 1px;
}
.term-desc { font-size: 0.85rem; color: var(--fg-muted); }
.card-links { display: flex; gap: 8px; flex-wrap: wrap; }
.btn-link {
  font-family: var(--font-m); font-size: 0.7rem; color: var(--accent);
  text-decoration: none; border: 1px solid var(--accent);
  padding: 5px 12px; border-radius: 4px; opacity: 0.85; transition: opacity 0.15s;
}
.btn-link:hover { opacity: 1; }
.btn-gh { color: var(--fg-muted); border-color: var(--border); }
.btn-gh:hover { color: var(--fg); border-color: rgba(255,255,255,0.25); }
.footer {
  text-align: center; font-family: var(--font-m); font-size: 0.72rem;
  color: var(--fg-dim); padding: 24px 16px 40px;
}
.footer a { color: var(--fg-muted); }
"""


def render_html(papers: list[dict], content_list: list[dict], date_str: str) -> str:
    content_map = {item["id"]: item for item in content_list}
    newsclaw_url = f"{NEWSCLAW_BASE}/{date_str.replace('-', '/')}/{SLUG}/"

    def tag_span(tag: str) -> str:
        color = TAG_COLORS.get(tag.upper(), "#6b7280")
        return f'<span class="tag" style="background:{color}">{tag}</span>'

    def esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    cards = []
    for i, p in enumerate(papers, 1):
        c = content_map.get(i, {})
        overview = esc(c.get("overview", ""))
        points = c.get("points", [])
        terms = c.get("terms", [])

        tags_html = "".join(tag_span(t) for t in p["tags"])

        points_html = "\n".join(f"<li>{esc(pt)}</li>" for pt in points)
        points_section = (
            f'<div class="section-block points">'
            f'<div class="section-label">🎯 論文のポイント</div>'
            f'<ul class="points-list">{points_html}</ul></div>'
            if points else ""
        )

        terms_html = ""
        for t in terms:
            terms_html += (
                f'<div class="term-item">'
                f'<span class="term-word">{esc(t.get("word", ""))}</span>'
                f'<span class="term-desc">{esc(t.get("desc", ""))}</span>'
                f'</div>'
            )
        terms_section = (
            f'<div class="section-block terms">'
            f'<div class="section-label">📚 用語解説</div>'
            f'<div class="terms-grid">{terms_html}</div></div>'
            if terms else ""
        )

        links = f'<a class="btn-link" href="{p["arxiv_url"]}" target="_blank">📄 arxiv</a>'
        if p.get("github_url"):
            links += f' <a class="btn-link btn-gh" href="{p["github_url"]}" target="_blank">⌥ GitHub</a>'

        cards.append(f"""
<article class="paper-card">
  <div class="card-top">
    <div class="vote-block">
      <span class="vote-arrow">▲</span>
      <span class="vote-num">{p['votes']}</span>
      <span class="rank">#{i}</span>
    </div>
    <div>
      <h2 class="paper-title"><a href="{p['arxiv_url']}" target="_blank">{esc(p['title'])}</a></h2>
      <div class="tags">{tags_html}</div>
    </div>
  </div>
  <div class="card-sections">
    <div class="section-block">
      <div class="section-label">📝 概要</div>
      <p>{overview}</p>
    </div>
    {points_section}
    {terms_section}
    <div class="card-links">{links}</div>
  </div>
</article>""")

    body = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI論文ダイジェスト — {date_str}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400&display=swap">
<style>{CSS}</style>
</head>
<body>
<header class="site-header">
  <div class="header-inner">
    <div style="flex:1;min-width:180px">
      <div class="header-eyebrow">ArXiv Trend Digest</div>
      <div class="header-title">今日のAI論文ダイジェスト</div>
    </div>
    <span class="header-meta">{date_str}</span>
    <a class="header-link" href="{newsclaw_url}" target="_blank">NewsClaw ↗</a>
  </div>
</header>
<main class="main">
  <p class="section-label-top">トップ10論文 — upvote 順 — 概要・ポイント・用語解説付き</p>
  {body}
</main>
<footer class="footer">
  <p>ソース: <a href="{newsclaw_url}" target="_blank">NewsClaw</a> / 生成: Claude ({MODEL})</p>
</footer>
</body>
</html>"""


# ── Slack post ────────────────────────────────────────────────────────────────

def build_slack_blocks(
    papers: list[dict],
    content_list: list[dict],
    date_str: str,
    page_url: str,
) -> list[dict]:
    content_map = {item["id"]: item for item in content_list}
    newsclaw_url = f"{NEWSCLAW_BASE}/{date_str.replace('-', '/')}/{SLUG}/"

    blocks: list[dict] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📰 今日のAI論文ダイジェスト — {date_str}"},
        },
        {"type": "divider"},
    ]

    for i, p in enumerate(papers, 1):
        c = content_map.get(i, {})
        short = c.get("short", p["title"][:40])
        tags = "  ".join(f"`{t}`" for t in p["tags"]) if p["tags"] else ""
        links = f"<{p['arxiv_url']}|📄 arxiv>"
        if p.get("github_url"):
            links += f"  ·  <{p['github_url']}|⌥ GitHub>"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*{i}. {p['title']}*  ▲{p['votes']}\n"
                    f"{tags}\n"
                    f"{short}\n"
                    f"{links}"
                ),
            },
        })

    blocks.append({"type": "divider"})
    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f":memo: *概要・ポイント・用語解説* → <{page_url}|詳細ページを開く>\n"
                f":link: 全論文リスト → <{newsclaw_url}|NewsClaw>"
            ),
        },
    })

    return blocks


def post_to_slack(webhook_url: str, blocks: list[dict]) -> None:
    resp = requests.post(
        webhook_url,
        json={"blocks": blocks},
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    resp.raise_for_status()


# ── GitHub Pages URL helper ────────────────────────────────────────────────────

def pages_url(date_str: str) -> str:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" in repo:
        owner, name = repo.split("/", 1)
        return f"https://{owner}.github.io/{name}/{date_str}.html"
    return f"https://example.github.io/arxiv-digest/{date_str}.html"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        sys.exit("Error: SLACK_WEBHOOK_URL is not set.")
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Error: GEMINI_API_KEY is not set.")

    url, date_str = today_url()
    print(f"Fetching: {url}")

    try:
        papers = fetch_papers(url)
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            print(f"No digest page for {date_str}, skipping.")
            return
        raise

    if not papers:
        print("No papers found.")
        return

    print(f"Found {len(papers)} papers. Fetching abstracts...")
    abstracts = fetch_abstracts(papers)
    print(f"Fetched {len(abstracts)} abstracts. Generating content via Claude API...")

    content_list = generate_content(papers, abstracts)
    print(f"Generated content for {len(content_list)} papers. Rendering HTML...")

    html = render_html(papers, content_list, date_str)
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)

    html_path = docs_dir / f"{date_str}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path}")

    page_url = pages_url(date_str)
    index_html = (
        f'<!DOCTYPE html><meta charset="UTF-8">'
        f'<meta http-equiv="refresh" content="0;url={date_str}.html">'
        f'<script>location.href="{date_str}.html"</script>'
        f'<p><a href="{date_str}.html">今日のダイジェスト ({date_str})</a></p>'
    )
    (docs_dir / "index.html").write_text(index_html, encoding="utf-8")
    print("Updated docs/index.html")

    print("Posting to Slack...")
    blocks = build_slack_blocks(papers, content_list, date_str, page_url)
    post_to_slack(webhook, blocks)
    print(f"Done. {len(papers)} papers posted. Page: {page_url}")


if __name__ == "__main__":
    main()
