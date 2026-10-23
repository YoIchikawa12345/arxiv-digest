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

ARXIV_QUERY = "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV"


# ── Fetch ─────────────────────────────────────────────────────────────────────────────────

def fetch_candidates() -> list[dict]:
    """arXiv APIからAI/ML論文を取得し、投稿日時順で返す。"""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": ARXIV_QUERY,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": 100,
    }

    resp = None
    for attempt in range(3):
        try:
            print(f"  Querying arXiv API (attempt {attempt + 1})...")
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                print(f"  Warning: {e}")
                return []
            wait = 20 * (attempt + 1)
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    if resp is None:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)

    papers = []
    seen = set()
    for entry in root.findall("atom:entry", ns):
        id_el = entry.find("atom:id", ns)
        if id_el is None:
            continue
        arxiv_id = id_el.text.split("abs/")[-1].strip()
        if arxiv_id in seen:
            continue
        seen.add(arxiv_id)

        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)

        title = (title_el.text if title_el is not None else "").strip().replace("\n", " ")
        abstract = (summary_el.text if summary_el is not None else "").strip()[:500]
        published = ((published_el.text if published_el is not None else "")[:10])
        year = published[:4]

        categories = [c.get("term", "") for c in entry.findall("atom:category", ns)]
        papers.append({
            "paperId": arxiv_id,
            "arxivId": arxiv_id,
            "title": title,
            "year": year,
            "published": published,
            "categories": categories,
            "citationCount": 0,
            "influentialCitationCount": 0,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })

    papers.sort(key=lambda p: p.get("published", ""), reverse=True)
    print(f"  Total candidates: {len(papers)}")
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
        if p.get("paperId") in sent_ids:
            continue
        result.append(p)
        if len(result) >= PAPERS_PER_RUN:
            break
    return result


# ── Summarize ─────────────────────────────────────────────────────────────────────────────

SUMMARIZE_BATCH = 10  # Gemini に一度に送る論文数


def _summarize_batch(client, papers: list[dict]) -> list[str]:
    parts = [
        f"[{i}] {p['title']} ({p['year']})\n{p['abstract']}"
        for i, p in enumerate(papers)
    ]
    prompt = (
        "以下のAI・機械学習論文を、日本語でわかりやすく解説してください。\n"
        "対象読者はAIや機械学習をあまり知らない初心者です。\n\n"
        "各論文について、以下の点を含めて4～6文程度で解説してください：\n"
        "1. どんな課題・問題を解こうとしているか（背景）\n"
        "2. どんな手法やアルゴリズムを使っているか（内容）\n"
        "3. 実験でどんな結果が得られたか（成果）\n\n"
        "注意事項：\n"
        "- 「歴史的な論文」「分野を変えた」「金字塔」などの評価コメントは一切書かない\n"
        "- 論文の社会的意義や影響の説明も不要\n"
        "- 専門用語は必ずかっこ内で平易に説明する\n"
        "- 初めて読む人が「何をしている論文か」をイメージできる具体的な表現を使う\n\n"
        "必ず以下のJSON形式のみで回答:\n"
        '{"0": "解説文", "1": "解説文", ...}\n\n'
        + "\n\n".join(parts)
    )
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
            raw = resp.text.strip()
            m = re.search(r'\{[\s\S]*\}', raw)
            if m:
                data = json.loads(m.group())
                return [data.get(str(i), "") for i in range(len(papers))]
        except Exception as e:
            if attempt == 2:
                print(f"  Gemini error: {e}")
                return [""] * len(papers)
            time.sleep(30 * (attempt + 1))
    return [""] * len(papers)


def summarize(api_key: str, papers: list[dict]) -> list[str]:
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
      html += '<article class="card">' +
        '<div class="card-meta">' +
        '<span class="date">📅 ' + esc(p.published||p.year||'') + '</span>' +
        '<span class="card-cat">' + label + '</span>' +
        '<span>追加: ' + esc(p.addedDate||'') + '</span>' +
        '</div>' +
        '<h2 class="card-title"><a href="' + esc(p.url||'#') + '" target="_blank" rel="noopener">' + esc(p.title||'') + '</a></h2>' +
        '<p class="card-summary">' + esc(p.summary||'（要約なし）') + '</p>' +
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
    date_str = datetime.now(JST).strftime("%Y年%m月%d日")
    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📚 AI論文まとめ 新着 — {date_str}"},
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

    print("Fetching arXiv candidates...")
    candidates = fetch_candidates()

    existing = load_papers()
    sent_ids = {p["paperId"] for p in existing}

    new_papers = pick_new(candidates, sent_ids)
    if not new_papers:
        print("No new papers to add.")
        save_papers(existing)
        html = generate_index_html(existing)
        (DOCS_DIR / "index.html").write_text(html, encoding="utf-8")
        return

    print(f"Summarizing {len(new_papers)} new papers...")
    summaries = summarize(api_key, new_papers)

    today = datetime.now(JST).strftime("%Y-%m-%d")
    for p, s in zip(new_papers, summaries):
        p["summary"] = s
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
