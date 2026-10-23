#!/usr/bin/env python3
"""
arXiv 最新AI論文ダイジェスト
- arXiv APIから最新AI/ML論文を取得
- 送信済み論文を除外し、毎回PAPERS_PER_RUN件ずつ追加
- docs/papers.json に蓄積、docs/index.html を再生成
- 新規追加論文をSlackに通知
"""

import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from google import genai

JST = timezone(timedelta(hours=9))
GEMINI_MODEL = "gemini-3.8-flash"
PAPERS_PER_RUN = 30
DOCS_DIR = Path("docs")
PAPERS_JSON = DOCS_DIR / "papers.json"
PAGES_URL = "https://yoichikawa12345.github.io/arxiv-digest/"

SS_QUERIES = [
    "deep learning neural network",
    "large language model transformer",
    "machine learning optimization",
    "computer vision image recognition",
    "natural language processing",
    "reinforcement learning policy",
    "generative model diffusion",
    "graph neural network",
]

# ── Fetch ─────────────────────────────────────────────────────────────────────────────────

def fetch_candidates(ss_api_key: str = "") -> list[dict]:
    """Semantic Scholar 検索APIで高影響力論文を取得し、引用数順で返す。
    API key があればそれを使う。なければ非認証で試みる（レート制限あり）。"""
    headers = {"x-api-key": ss_api_key} if ss_api_key else {}
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "paperId,title,abstract,citationCount,influentialCitationCount,year,externalIds"

    seen = set()
    papers = []

    for query in SS_QUERIES:
        params = {
            "query": query,
            "fields": fields,
            "fieldsOfStudy": "Computer Science",
            "limit": 100,
        }
        for attempt in range(4):
            try:
                print(f"  Querying SS: {query!r} (attempt {attempt+1})...")
                resp = requests.get(url, params=params, headers=headers, timeout=20)
                if resp.status_code == 429:
                    wait = 20 * (attempt + 1)
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                for p in resp.json().get("data", []):
                    pid = p.get("paperId", "")
                    if not pid or pid in seen:
                        continue
                    seen.add(pid)
                    arxiv_id = (p.get("externalIds") or {}).get("ArXiv", "")
                    papers.append({
                        "paperId": pid,
                        "arxivId": arxiv_id,
                        "title": p.get("title", ""),
                        "year": p.get("year", ""),
                        "published": str(p.get("year", "")),
                        "categories": [],
                        "citationCount": p.get("citationCount") or 0,
                        "influentialCitationCount": p.get("influentialCitationCount") or 0,
                        "abstract": (p.get("abstract") or "")[:500],
                        "url": "https://arxiv.org/abs/" + arxiv_id if arxiv_id
                               else "https://www.semanticscholar.org/paper/" + pid,
                    })
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  Warning ({query!r}): {e}")
                else:
                    time.sleep(5 * (attempt + 1))
        time.sleep(1)  # polite delay

    papers.sort(
        key=lambda p: (p.get("influentialCitationCount", 0), p.get("citationCount", 0)),
        reverse=True,
    )
    top = papers[0] if papers else {}
    print(f"  Total candidates: {len(papers)}, top influential={top.get('influentialCitationCount', 0)}: {top.get('title','')[:50]}")
    return papers

def load_papers() -> list[dict]:
    if PAPERS_JSON.exists():
        return json.loads(PAPERS_JSON.read_text(encoding="utf-8-sig"))
    return []

def save_papers(papers: list[dict]) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    PAPERS_JSON.write_text(
        json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def pick_new(candidates: list[dict], sent_ids: set) -> list[dict]:
    result = []
    for p in candidates:
        uid = p.get("paperId", "")
        if uid in sent_ids:
            continue
        result.append(p)
        if len(result) >= PAPERS_PER_RUN:
            break
    return result

# ── Summarize ─────────────────────────────────────────────────────────────────────────────

SUMMARIZE_BATCH = 10  # Gemini に一度に送る論文数

class GeminiQuotaExceeded(Exception):
    pass

def _summarize_batch(client, papers: list[dict]) -> list[dict]:
    parts = [
        f"[{i}] {p['title']} ({p['year']})\n{p['abstract']}"
        for i, p in enumerate(papers)
    ]
    prompt = (
        "以下のAI・機械学習論文を、日本語でわかりやすく解説してください。\n"
        "対象読者はAIや機械学習をあまり知らない初心者です。\n\n"
        "各論文について以下を返してください：\n"
        "1. summary: 4～6文程度の解説（背景・手法・成果）\n"
        "   - 「歴史的な論文」「分野を変えた」などの評価コメントは書かない\n"
        "   - 初めて読　人がイメージできる具体的な表現を使う\n"
        "2. terms: 論文中の重要な専門用語を3～5個ピックアップし、平易な日本語で1文ずつ解説\n\n"
        "必ず以下のJSON形式のみで回答:\n"
        '{"0": {"summary": "解説文", "terms": {"用語1": "説明", "用語2": "説明"}}, "1": {...}, ...}\n\n'
        + "\n\n".join(parts)
    )
    empty: dict = {"summary": "", "terms": {}}
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = resp.text.strip()
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                data = json.loads(m.group())
                return [data.get(str(i), empty) for i in range(len(papers))]
        except Exception as e:
            err = str(e).lower()
            if any(kw in err for kw in ("quota", "resource_exhausted", "429", "rate limit")):
                raise GeminiQuotaExceeded(str(e))
            if attempt == 2:
                print(f"  Gemini error: {e}")
                return [empty] * len(papers)
            time.sleep(30 * (attempt + 1))
    return [empty] * len(papers)

def summarize(api_key: str, papers: list[dict]) -> list[dict]:
    client = genai.Client(api_key=api_key)
    results = []
    for i in range(0, len(papers), SUMMARIZE_BATCH):
        batch = papers[i:i + SUMMARIZE_BATCH]
        print(f"  Summarizing batch {i // SUMMARIZE_BATCH + 1} ({len(batch)} papers)...")
        results.extend(_summarize_batch(client, batch))
        if i + SUMMARIZE_BATCH < len(papers):
            time.sleep(5)
    return results

# ── HTML ─────────────────────────────────────────────────────────────────────────────────────

def generate_index_html(all_papers: list[dict]) -> str:
    count = len(all_papers)
    updated = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AI論文まとめ</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap">
<style>
:root {{
  --bg:#fff;--bg2:#f7f8fa;--fg:#1a1a1a;--fg2:#555;--fg3:#888;
  --border:#e0e3e8;--accent:#3563e9;--accent-light:#e8eeff;--link:#3563e9;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme=light]) {{
    --bg:#0f1117;--bg2:#181c25;--fg:#e8eaf0;--fg2:#a0a8b8;--fg3:#6b7280;
    --border:#2a2f3e;--accent:#6b9aff;--accent-light:#1a2340;--link:#6b9aff;
    color-scheme:dark;
  }}
}}
:root[data-theme=dark] {{
  --bg:#0f1117;--bg2:#181c25;--fg:#e8eaf0;--fg2:#a0a8b8;--fg3:#6b7280;
  --border:#2a2f3e;--accent:#6b9aff;--accent-light:#1a2340;--link:#6b9aff;
  color-scheme:dark;
}}
*,*::before,*::after{{box-sizing:border-box}}
body{{
  background:var(--bg);color:var(--fg);
  font-family:'Noto Sans JP',sans-serif;font-size:15px;line-height:1.8;
  margin:0;
  padding-left:max(16px,env(safe-area-inset-left,0px));
  padding-right:max(16px,env(safe-area-inset-right,0px));
  padding-block:0 64px;
}}
.container{{max-width:760px;margin:0 auto}}
.site-header{{padding-block:40px 24px;border-bottom:1px solid var(--border);margin-bottom:24px}}
.site-header h1{{margin:0 0 4px;font-size:24px;font-weight:700}}
.site-header p{{margin:0;font-size:13px;color:var(--fg3)}}
.search-wrap{{margin-bottom:16px}}
#searchInput{{
  width:100%;padding:10px 14px;font-size:15px;
  border:1px solid var(--border);border-radius:8px;
  background:var(--bg2);color:var(--fg);outline:none;font-family:inherit;
}}
#searchInput:focus{{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-light)}}
.tabs{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}}
.tab{{
  padding:6px 14px;border-radius:20px;border:1px solid var(--border);
  background:var(--bg2);color:var(--fg2);font-size:13px;font-weight:500;
  cursor:pointer;font-family:inherit;transition:background .15s,color .15s;
}}
.tab:hover,.tab.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.top-bar{{
  display:flex;justify-content:space-between;align-items:center;
  flex-wrap:wrap;gap:8px;margin-bottom:12px;font-size:13px;color:var(--fg3);
}}
.top-bar strong{{color:var(--fg)}}
.pagination{{display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
.page-btn{{
  min-width:36px;height:36px;padding:0 10px;border-radius:6px;
  border:1px solid var(--border);background:var(--bg2);color:var(--fg2);
  font-size:13px;cursor:pointer;font-family:inherit;
}}
.page-btn:hover,.page-btn.active{{background:var(--accent);color:#fff;border-color:var(--accent)}}
.bottom-bar{{display:flex;justify-content:flex-end;margin-top:20px}}
.card{{border:1px solid var(--border);border-radius:10px;padding:20px 24px;margin-bottom:16px;background:var(--bg)}}
.card-meta{{display:flex;flex-wrap:wrap;gap:10px;font-size:12px;color:var(--fg3);margin-bottom:8px}}
.date{{color:var(--accent);font-weight:700}}
.card-cat{{background:var(--accent-light);color:var(--accent);padding:1px 8px;border-radius:20px;font-size:11px;font-weight:700}}
.card-title{{margin:0 0 10px;font-size:16px;font-weight:700;line-height:1.4}}
.card-title a{{color:var(--link);text-decoration:none}}
.card-title a:hover{{text-decoration:underline}}
.card-summary{{margin:0;font-size:14px;color:var(--fg2);line-height:1.7}}
.card-terms{{margin-top:12px;padding-top:12px;border-top:1px dashed var(--border)}}
.card-terms-label{{font-size:11px;font-weight:700;color:var(--fg3);text-transform:uppercase;letter-spacing:.05em;margin-bottom:6px}}
.card-terms dl{{margin:0;display:flex;flex-direction:column;gap:3px}}
.card-terms dt{{font-size:12px;font-weight:700;color:var(--accent);display:inline}}
.card-terms dd{{font-size:12px;color:var(--fg2);margin:0;display:inline}}
.no-results{{text-align:center;padding:64px 0;color:var(--fg3)}}
.loading{{text-align:center;padding:40px 0;color:var(--fg3)}}
</style>
</head>
<body>
<div class="container">
  <header class="site-header">
    <h1>AI論文まとめ</h1>
    <p>arXivの最新AI・機械学習論文の日本語要約集 &nbsp;·&nbsp; 全 {count} 件 &nbsp;·&nbsp; 最終更新 {updated}</p>
  </header>
  <div class="search-wrap">
    <input type="search" id="searchInput" placeholder="タイトル・要約を検索…" autocomplete="off">
  </div>
  <div class="tabs" role="tablist">
    <button class="tab active" onclick="setTab(this,'all')">すべて</button>
    <button class="tab" onclick="setTab(this,'cs.AI')">🤖 AI・推論</button>
    <button class="tab" onclick="setTab(this,'cs.LG')">📊 機械学習</button>
    <button class="tab" onclick="setTab(this,'cs.CL')">💬 自然言語処理</button>
    <button class="tab" onclick="setTab(this,'cs.CV')">🖼 画像・Vision</button>
  </div>
  <div class="top-bar">
    <span id="statsText"></span>
    <div class="pagination" id="paginTop"></div>
  </div>
  <div id="cards"><div class="loading">読み込み中…</div></div>
  <div class="bottom-bar">
    <div class="pagination" id="paginBottom"></div>
  </div>
</div>
<script>
(function() {{
  var PER_PAGE = 10, page = 1, cat = 'all', q = '', PAPERS = [];
  var CAT_LABELS = {{'cs.AI':'AI・推論','cs.LG':'機械学習','cs.CL':'自然言語処理','cs.CV':'画像・Vision'}};
  function getCat(p) {{
    var c = (p.categories || [])[0];
    if (c) return c;
    var t = ((p.title||'') + ' ' + (p.summary||'')).toLowerCase();
    if (/language model|llm|nlp|translation|tokeniz|speech|chat|dialogue|prompt/.test(t)) return 'cs.CL';
    if (/image|vision|visual|pixel|video|3d|segmentation|detection|recognition|diffusion/.test(t)) return 'cs.CV';
    if (/reinforcement|reward|policy|agent|planning|reasoning|logic|knowledge/.test(t)) return 'cs.AI';
    return 'cs.LG';
  }}
  function esc(s) {{
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}
  function filtered() {{
    return PAPERS.filter(function(p) {{
      if (cat !== 'all' && getCat(p) !== cat) return false;
      if (q && ((p.title||'') + ' ' + (p.summary||'')).toLowerCase().indexOf(q) < 0) return false;
      return true;
    }});
  }}
  function renderPagin(total, id) {{
    var el = document.getElementById(id);
    var pages = Math.ceil(total / PER_PAGE);
    if (pages <= 1) {{ el.innerHTML = ''; return; }}
    var h = '';
    if (page > 1) h += '<button class="page-btn" onclick="go(' + (page-1) + ')">← 前へ</button>';
    for (var i = 1; i <= pages; i++) {{
      if (i === 1 || i === pages || Math.abs(i-page) <= 2)
        h += '<button class="page-btn' + (i===page?' active':'') + '" onclick="go(' + i + ')">' + i + '</button>';
      else if (Math.abs(i-page) === 3)
        h += '<span style="color:var(--fg3);padding:0 4px">…</span>';
    }}
    if (page < pages) h += '<button class="page-btn" onclick="go(' + (page+1) + ')">次へ →</button>';
    el.innerHTML = h;
  }}
  function render() {{
    var f = filtered(), total = f.length;
    var slice = f.slice((page-1)*PER_PAGE, page*PER_PAGE);
    document.getElementById('statsText').innerHTML =
      '<strong>' + total + '</strong> 件 / 全 <strong>' + PAPERS.length + '</strong> 件';
    renderPagin(total,'paginTop');
    renderPagin(total,'paginBottom');
    if (!slice.length) {{
      document.getElementById('cards').innerHTML = '<div class="no-results">該当する論文が見つかりませんでした。</div>';
      return;
    }}
    var html = '';
    for (var j = 0; j < slice.length; j++) {{
      var p = slice[j], c = getCat(p), label = CAT_LABELS[c] || c;
      var terms = p.terms || {{}}, termKeys = Object.keys(terms), termsHtml = '';
      if (termKeys.length) {{
        termsHtml = '<div class="card-terms"><div class="card-terms-label">📖 用語解説</div><dl>' +
          termKeys.map(function(k) {{ return '<dt>' + esc(k) + '</dt><dd>: ' + esc(terms[k]) + '</dd>'; }}).join('') +
          '</dl></div>';
      }}
      html += '<article class="card">' +
        '<div class="card-meta">' +
        '<span class="date">📅 ' + esc(p.published||p.year||'') + '</span>' +
        '<span class="card-cat">' + label + '</span>' +
        '<span>追加: ' + esc(p.addedDate||'') + '</span>' +
        '</div>' +
        '<h2 class="card-title"><a href="' + esc(p.url||'#') + '" target="_blank" rel="noopener">' + esc(p.title||'') + '</a></h2>' +
        '<p class="card-summary">' + esc(p.summary||'（要約なし）') + '</p>' +
        termsHtml +
        '</article>';
    }}
    document.getElementById('cards').innerHTML = html;
  }}
  window.go = function(n) {{ page = n; render(); window.scrollTo(0,0); }};
  window.setTab = function(btn, c) {{
    document.querySelectorAll('.tab').forEach(function(t) {{ t.classList.remove('active'); }});
    btn.classList.add('active');
    cat = c; page = 1; render();
  }};
  document.getElementById('searchInput').addEventListener('input', function(e) {{
    q = e.target.value.toLowerCase().trim(); page = 1; render();
  }});
  fetch('papers.json')
    .then(function(r) {{ return r.json(); }})
    .then(function(data) {{ PAPERS = data; render(); }})
    .catch(function() {{
      document.getElementById('cards').innerHTML = '<div class="no-results">データの読み込みに失敗しました。</div>';
    }});
}})();
</script>
</body>
</html>"""

# ── Slack ──────────────────────────────────────────────────────────────────────────────────

def post_to_slack(webhook_url: str, new_papers: list[dict]) -> None:
    date_str = datetime.now(JST).strftime("%Y噴%m月%d日")
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📚 AI論文まとめ 新着 — {date_str}"},
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"<{PAGES_URL}|📄 まとめページで全論文を見る>"}],
        },
        {"type": "divider"},
    ]
    for p in new_papers:
        summary = p.get("summary", "（要約なし）")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join([
                    f"*{p['title']}* ({p.get('year','')})",
                    f"📅 投稿: {p.get('published', p.get('year',''))}",
                    summary,
                ]),
            },
        })
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"<{p['url']}|論文を読む>"}],
        })
        blocks.append({"type": "divider"})

    resp = requests.post(webhook_url, json={"blocks": blocks[:50]}, timeout=15)
    resp.raise_for_status()
    print("Slack post OK.")

# ── Main ──────────────────────────────────────────────────────────────────────────────────

def main():
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("Error: GEMINI_API_KEY is not set.")

    ss_api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    print("Fetching Semantic Scholar candidates...")
    candidates = fetch_candidates(ss_api_key)

    existing = load_papers()
    sent_ids = {p.get("paperId", "") for p in existing}

    new_papers = pick_new(candidates, sent_ids)
    if not new_papers:
        print("No new papers to add.")
        save_papers(existing)
        html = generate_index_html(existing)
        (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
        return

    print(f"Summarizing {len(new_papers)} new papers...")
    try:
        summaries = summarize(api_key, new_papers)
    except GeminiQuotaExceeded as e:
        print(f"Gemini quota exceeded, skipping update: {e}")
        return

    today = datetime.now(JST).strftime("%Y-%m-%d")
    for p, s in zip(new_papers, summaries):
        p["summary"] = s.get("summary", "") if isinstance(s, dict) else s
        p["terms"] = s.get("terms", {}) if isinstance(s, dict) else {}
        p["addedDate"] = today
        del p["abstract"]

    all_papers = existing + new_papers
    save_papers(all_papers)
    print(f"papers.json updated: {len(all_papers)} papers total.")

    html = generate_index_html(all_papers)
    (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
    print("docs/index.html generated.")

    if webhook:
        try:
            post_to_slack(webhook, new_papers[:15])
        except Exception as e:
            print(f"Slack post failed (non-fatal): {e}")
    else:
        print("SLACK_WEBHOOK_URL not set, skipping Slack post.")

if __name__ == "__main__":
    main()
