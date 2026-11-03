"""產生三個實驗的 委託.html。

委託的文字（欄位說明、提醒、步驟、交付清單）都寫在下面的 PAGES。
改了委託內容、版面，或改了 data/ 裡委託用的 CSV，就重跑：

    uv run python tools/build_briefs.py

CSV 會原樣嵌進 HTML，學生直接用瀏覽器打開就能瀏覽資料。
各實驗的 tests/test_brief.py 會檢查嵌入的資料跟 CSV 一致。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIN = ROOT / "linear-regression/src/add_linear_regression/experiment"
LOG = ROOT / "logistic-regression/src/add_logistic_regression/experiment"
NN = ROOT / "neural-networks/src/add_neural_networks/experiment"

STYLE = """<style>
  :root {
    --bg: #f4efe4;
    --paper: #fffdf8;
    --ink: #2b2a26;
    --muted: #6f6a5f;
    --line: #e2dac8;
    --leaf: #3f7d3a;
    --leaf-soft: #e3f0dc;
    --target: #1f5fa8;
    --target-soft: #dfeaf7;
    --stop: #b23a2e;
    --stop-soft: #f7e0dc;
    --ref: #8a8270;
    --ref-soft: #ece7dc;
    --stamp: #c0392b;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #1b1d1a;
      --paper: #252822;
      --ink: #ece8dd;
      --muted: #a39e91;
      --line: #3a3d36;
      --leaf: #8cc97f;
      --leaf-soft: #2c3b28;
      --target: #8ab8ec;
      --target-soft: #233246;
      --stop: #ef8b7f;
      --stop-soft: #43281f;
      --ref: #b2ab9b;
      --ref-soft: #33342e;
      --stamp: #ef8b7f;
    }
  }
  :root[data-theme="dark"] {
    --bg: #1b1d1a; --paper: #252822; --ink: #ece8dd; --muted: #a39e91; --line: #3a3d36;
    --leaf: #8cc97f; --leaf-soft: #2c3b28; --target: #8ab8ec; --target-soft: #233246;
    --stop: #ef8b7f; --stop-soft: #43281f; --ref: #b2ab9b; --ref-soft: #33342e; --stamp: #ef8b7f;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", "Heiti TC", sans-serif;
    line-height: 1.75;
    font-size: 16px;
  }
  .sheet {
    max-width: 880px;
    margin: 32px auto;
    padding: 0 16px;
  }
  .card {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 28px 32px;
    margin-bottom: 20px;
  }
  @media (max-width: 600px) { .card { padding: 20px 18px; } }

  /* 抬頭 */
  .hero { position: relative; overflow: hidden; display: flex; gap: 24px; align-items: center; }
  .hero svg { flex: 0 0 120px; width: 120px; height: 120px; }
  .hero .eyebrow { color: var(--muted); font-size: 14px; letter-spacing: .12em; margin: 0; }
  .hero h1 { margin: 4px 0 8px; font-size: 30px; line-height: 1.3; }
  .hero p { margin: 0; color: var(--muted); }
  .stamp {
    position: absolute; top: 18px; right: 22px;
    border: 3px solid var(--stamp); color: var(--stamp);
    border-radius: 8px; padding: 2px 12px;
    font-weight: 700; letter-spacing: .2em; transform: rotate(8deg); opacity: .8;
  }
  @media (max-width: 600px) {
    .hero { flex-direction: column; align-items: flex-start; }
    .hero svg { width: 88px; height: 88px; flex-basis: auto; }
    .stamp { top: 14px; right: 14px; }
  }

  .facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 22px; }
  .fact { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .fact b { display: block; font-size: 22px; }
  .fact span { color: var(--muted); font-size: 14px; }
  @media (max-width: 600px) { .facts { grid-template-columns: 1fr; } }

  h2 { font-size: 20px; margin: 0 0 14px; display: flex; align-items: center; gap: 10px; }
  h2 .no {
    display: inline-grid; place-items: center; width: 28px; height: 28px; border-radius: 50%;
    background: var(--leaf); color: var(--paper); font-size: 15px;
  }

  /* 目標 */
  .goal {
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
    background: var(--target-soft); border-radius: 12px; padding: 16px 18px;
  }
  .goal .arrow { font-size: 26px; color: var(--target); }
  .goal .pill { background: var(--paper); border: 2px solid var(--target); color: var(--target); }

  .pill {
    display: inline-block; border-radius: 999px; padding: 2px 12px; font-size: 14px;
    border: 1px solid transparent; white-space: nowrap;
  }
  .pill.use { background: var(--leaf-soft); color: var(--leaf); }
  .pill.target { background: var(--target-soft); color: var(--target); }
  .pill.stop { background: var(--stop-soft); color: var(--stop); }
  .pill.ref { background: var(--ref-soft); color: var(--ref); }

  /* 欄位 */
  .legend { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
  .cols { display: grid; gap: 10px; }
  .col {
    display: grid; grid-template-columns: 220px 1fr; gap: 14px; align-items: start;
    border: 1px solid var(--line); border-left: 5px solid var(--ref); border-radius: 10px; padding: 12px 14px;
  }
  .col.use { border-left-color: var(--leaf); }
  .col.target { border-left-color: var(--target); }
  .col.stop { border-left-color: var(--stop); }
  .col .name { font-weight: 700; }
  .col .name .pill { margin-top: 4px; font-weight: 400; }
  .col p { margin: 0; }
  @media (max-width: 600px) { .col { grid-template-columns: 1fr; gap: 6px; } }

  .warn {
    margin-top: 16px; border-radius: 12px; padding: 14px 16px;
    background: var(--stop-soft); color: var(--ink);
  }
  .warn strong { color: var(--stop); }

  /* 資料樣貌 */
  .scroll { overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; min-width: 640px; }
  th, td { padding: 8px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
  th { background: var(--ref-soft); font-weight: 600; text-align: center; }
  td:first-child, th:first-child { text-align: left; }
  tr:last-child td { border-bottom: 0; }
  th.t, td.t { background: var(--target-soft); }
  th.s, td.s { color: var(--muted); }
  td.blank { color: var(--stop); text-align: center; }
  .caption { color: var(--muted); font-size: 14px; margin-top: 10px; }
  th[data-i] { cursor: pointer; user-select: none; }
  th[data-i]:hover { color: var(--leaf); }
  .tools { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; font-size: 14px; }
  .tools select {
    font: inherit; color: var(--ink); background: var(--paper);
    border: 1px solid var(--line); border-radius: 8px; padding: 4px 8px; margin-left: 6px;
  }
  .tools .chk { display: flex; align-items: center; gap: 6px; cursor: pointer; }
  .tools .chk input { accent-color: var(--leaf); }
  .count { color: var(--muted); margin-left: auto; }
  .pager { display: flex; justify-content: center; align-items: center; gap: 14px; margin-top: 12px; font-size: 14px; }
  .pager button {
    font: inherit; color: var(--ink); background: var(--paper); cursor: pointer;
    border: 1px solid var(--line); border-radius: 8px; padding: 4px 12px;
  }
  .pager button:disabled { opacity: .4; cursor: default; }

  /* 步驟 */
  .steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; counter-reset: s; }
  .step { border: 1px solid var(--line); border-radius: 12px; padding: 14px; position: relative; }
  .step .k { color: var(--muted); font-size: 13px; }
  .step b { display: block; font-size: 18px; margin: 2px 0 6px; }
  .step p { margin: 0; font-size: 14px; }
  .step.me { border: 2px dashed var(--stamp); }
  .step.me .k { color: var(--stamp); }
  @media (max-width: 720px) { .steps { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 440px) { .steps { grid-template-columns: 1fr; } }

  .rule { margin: 16px 0 0; padding: 12px 16px; border-radius: 10px; background: var(--leaf-soft); }

  /* 繳交清單 */
  .checklist { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
  .checklist label {
    display: flex; gap: 12px; align-items: flex-start; cursor: pointer;
    border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px;
  }
  .checklist input { margin-top: 6px; width: 18px; height: 18px; accent-color: var(--leaf); flex: none; }
  .checklist input:checked + span { color: var(--muted); text-decoration: line-through; }

  code {
    font-family: "Cascadia Code", Consolas, monospace; font-size: .92em;
    background: var(--ref-soft); padding: 1px 6px; border-radius: 6px;
  }
  .foot { text-align: center; color: var(--muted); font-size: 13px; margin: 8px 0 40px; }
  .intro { margin: -4px 0 14px; color: var(--muted); }
  .card > .rule { margin-bottom: 0; }
</style>"""

SCRIPT = r"""<script>
(function () {
  var CFG = JSON.parse(document.getElementById("cfg").textContent);
  var text = document.getElementById("csv").textContent.trim();
  var lines = text.split(/\r?\n/);
  var header = lines[0].split(",");
  var data = lines.slice(1).map(function (l) { return l.split(","); });
  var PER_PAGE = 20;
  var sortCol = -1, sortDir = 1, page = 0;

  function cmpVal(x, y) {
    if (x === "" && y === "") return 0;
    if (x === "") return 1;
    if (y === "") return -1;
    var nx = Number(x), ny = Number(y);
    if (!isNaN(nx) && !isNaN(ny)) return nx - ny;
    return x < y ? -1 : x > y ? 1 : 0;
  }

  // 顯示順序：要預測的欄放最後，其他照 CSV 的順序。
  var order = header.map(function (h, i) { return i; })
    .filter(function (i) { return i !== CFG.target; }).concat([CFG.target]);

  var head = document.getElementById("head");
  order.forEach(function (i) {
    var th = document.createElement("th");
    th.dataset.i = i;
    if (i === CFG.target) th.className = "t";
    th.addEventListener("click", function () {
      sortDir = sortCol === i ? -sortDir : 1;
      sortCol = i; page = 0; render();
    });
    head.appendChild(th);
  });

  var filter = document.getElementById("filter");
  document.getElementById("filterLabel").firstChild.textContent = header[CFG.filter] + " ";
  var values = [];
  data.forEach(function (r) { if (values.indexOf(r[CFG.filter]) < 0) values.push(r[CFG.filter]); });
  values.sort(cmpVal).forEach(function (v) {
    var o = document.createElement("option");
    o.value = v; o.textContent = v === "" ? "空白" : v;
    filter.appendChild(o);
  });
  filter.addEventListener("change", function () { page = 0; render(); });

  var blankOnly = document.getElementById("blankOnly");
  var hasBlank = data.some(function (r) { return r.indexOf("") >= 0; });
  if (!hasBlank) blankOnly.parentNode.style.display = "none";
  blankOnly.addEventListener("change", function () { page = 0; render(); });
  document.getElementById("prev").addEventListener("click", function () { page--; render(); });
  document.getElementById("next").addEventListener("click", function () { page++; render(); });

  function render() {
    var rows = data.filter(function (r) {
      if (filter.value && r[CFG.filter] !== filter.value) return false;
      if (blankOnly.checked && r.indexOf("") < 0) return false;
      return true;
    });
    if (sortCol >= 0) rows = rows.slice().sort(function (a, b) {
      var x = a[sortCol], y = b[sortCol];
      if (x === "" || y === "") return cmpVal(x, y);
      return cmpVal(x, y) * sortDir;
    });
    var pages = Math.max(1, Math.ceil(rows.length / PER_PAGE));
    page = Math.min(Math.max(page, 0), pages - 1);

    Array.prototype.forEach.call(head.children, function (th) {
      var i = Number(th.dataset.i);
      th.textContent = header[i] + (i === sortCol ? (sortDir > 0 ? " ▲" : " ▼") : "");
    });

    var body = document.getElementById("rows");
    body.textContent = "";
    rows.slice(page * PER_PAGE, (page + 1) * PER_PAGE).forEach(function (r) {
      var tr = document.createElement("tr");
      order.forEach(function (i) {
        var v = r[i];
        var td = document.createElement("td");
        if (v === "") { td.textContent = "空白"; td.className = "blank"; }
        else { td.textContent = v; if (i === CFG.target) td.className = "t"; }
        tr.appendChild(td);
      });
      body.appendChild(tr);
    });

    document.getElementById("count").textContent = "共 " + rows.length + " 列（全部 " + data.length + " 列）";
    document.getElementById("page").textContent = "第 " + (page + 1) + " / " + pages + " 頁";
    document.getElementById("prev").disabled = page === 0;
    document.getElementById("next").disabled = page >= pages - 1;
  }
  render();
})();
</script>"""


def cols_html(cols, target):
    # 要預測的欄放最後，其他照 CSV 的順序。
    cols = [c for c in cols if c[0] != target] + [c for c in cols if c[0] == target]
    out = []
    for name, desc in cols:
        if name == target:
            out.append(
                f'      <div class="col target">\n        <div class="name">{name}<br><span class="pill target">要預測</span></div>\n        <p>{desc}</p>\n      </div>'
            )
        else:
            out.append(f'      <div class="col">\n        <div class="name">{name}</div>\n        <p>{desc}</p>\n      </div>')
    return "\n".join(out)


def page(p):
    csv_path = p["dir"] / "data" / p["csv"]
    csv_text = csv_path.read_text(encoding="utf-8").strip()
    assert "<" not in csv_text
    header = csv_text.splitlines()[0].split(",")
    target = header.index(p["target"])
    filt = header.index(p["filter"])
    facts = "\n".join(f'      <div class="fact"><b>{b}</b><span>{s}</span></div>' for b, s in p["facts"])
    steps = "\n".join(
        f'      <div class="step{" me" if i == 3 else ""}"><span class="k">{"最後一節 ・ 只有你寫" if i == 3 else f"第 {i + 1} 格 ・ 請 AI 寫"}</span><b>{t}</b><p>{d}</p></div>'
        for i, (t, d) in enumerate(p["steps"])
    )
    checks = "\n".join(f'      <li><label><input type="checkbox"><span>{c}</span></label></li>' for c in p["checks"])
    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{p["title"]}</title>
{STYLE}
</head>
<body>
<main class="sheet">

  <section class="card hero">
    <div class="stamp">委託</div>
{p["svg"]}
    <div>
      <p class="eyebrow">{p["eyebrow"]} ・ 實驗本做完後的案子</p>
      <h1>{p["h1"]}</h1>
      <p>{p["lead"]}</p>
    </div>
  </section>

  <section class="card">
    <div class="goal">
      <span>要估出</span>
      <span class="pill">{p["target"]}</span>
      <span class="arrow">←</span>
      <span>用哪些欄當特徵，由你決定</span>
    </div>
    <div class="facts">
{facts}
      <div class="fact"><b><code>data/{p["csv"]}</code></b><span>資料檔位置</span></div>
    </div>
  </section>

  <section class="card">
    <h2><span class="no">1</span>認識欄位</h2>
    <p class="intro">{p["row_note"]}</p>
    <div class="cols">
{cols_html(p["cols"], p["target"])}
    </div>
{p["notes"]}
  </section>

  <section class="card">
    <h2><span class="no">2</span>資料長這樣</h2>
    <div class="tools">
      <label id="filterLabel">欄位 <select id="filter"><option value="">全部</option></select></label>
      <label class="chk"><input type="checkbox" id="blankOnly">只看有空白的列</label>
      <span id="count" class="count"></span>
    </div>
    <div class="scroll">
      <table>
        <thead><tr id="head"></tr></thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
    <div class="pager">
      <button type="button" id="prev">‹ 上一頁</button>
      <span id="page"></span>
      <button type="button" id="next">下一頁 ›</button>
    </div>
    <p class="caption">內容就是 <code>data/{p["csv"]}</code>，按欄名可以排序，再按一次反過來。{p["data_note"]}</p>
  </section>

  <section class="card">
    <h2><span class="no">3</span>怎麼做</h2>
    <div class="steps">
{steps}
    </div>
    <p class="rule">在 <code>{p["notebook"]}</code> 裡完成。一次請 AI 寫一格，跑過、看懂，再進行下一格。程式用這個專案已經裝上的 {p["libs"]}。</p>
  </section>

  <section class="card">
    <h2><span class="no">4</span>做完要有</h2>
    <ul class="checklist">
{checks}
    </ul>
  </section>

  <p class="foot">讀完後，打開同資料夾的 <code>{p["notebook"]}</code> 開始做。</p>
</main>
<!-- 下面是 data/{p["csv"]} 的完整內容。資料檔改了，這裡也要重新貼上（測試會檢查兩邊一致）。 -->
<script id="csv" type="text/plain">
{csv_text}
</script>
<script id="cfg" type="application/json">{{"target": {target}, "filter": {filt}}}</script>
{SCRIPT}
</body>
</html>
"""


CABBAGE_SVG = """    <svg viewBox="0 0 120 120" aria-hidden="true">
      <circle cx="60" cy="64" r="44" fill="#9fcf7f"/>
      <path d="M60 22 C30 30 20 60 28 86 C40 70 46 52 60 40 Z" fill="#7dbb5c"/>
      <path d="M60 22 C90 30 100 60 92 86 C80 70 74 52 60 40 Z" fill="#6aa94b"/>
      <circle cx="60" cy="68" r="26" fill="#cfe9b4"/>
      <path d="M60 44 V96 M60 60 L44 50 M60 60 L76 50 M60 76 L42 68 M60 76 L78 68" stroke="#8ebf6c" stroke-width="3" fill="none" stroke-linecap="round"/>
    </svg>"""

BALL_SVG = """    <svg viewBox="0 0 120 120" aria-hidden="true">
      <circle cx="60" cy="60" r="46" fill="#f6f1e6" stroke="#cfc6b3" stroke-width="3"/>
      <path d="M30 26 C46 44 46 76 30 94" stroke="#c0392b" stroke-width="3" fill="none"/>
      <path d="M90 26 C74 44 74 76 90 94" stroke="#c0392b" stroke-width="3" fill="none"/>
      <path d="M33 36 l8 -3 M37 46 l8 -2 M38 56 l9 0 M38 66 l9 1 M36 76 l8 3 M33 86 l7 4" stroke="#c0392b" stroke-width="2.5" stroke-linecap="round"/>
      <path d="M87 36 l-8 -3 M83 46 l-8 -2 M82 56 l-9 0 M82 66 l-9 1 M84 76 l-8 3 M87 86 l-7 4" stroke="#c0392b" stroke-width="2.5" stroke-linecap="round"/>
    </svg>"""

FIT_SVG = """    <svg viewBox="0 0 120 120" aria-hidden="true">
      <circle cx="60" cy="60" r="48" fill="#f4dcc4"/>
      <rect x="34" y="56" width="52" height="8" rx="4" fill="#6f6a5f"/>
      <rect x="22" y="40" width="12" height="40" rx="4" fill="#e07b39"/>
      <rect x="86" y="40" width="12" height="40" rx="4" fill="#e07b39"/>
      <rect x="14" y="48" width="8" height="24" rx="3" fill="#c4622a"/>
      <rect x="98" y="48" width="8" height="24" rx="3" fill="#c4622a"/>
    </svg>"""

PAGES = [
    dict(
        dir=LIN, csv="taipei2_cabbage.csv", notebook="高麗菜估價.ipynb", title="高麗菜估價委託",
        svg=CABBAGE_SVG, eyebrow="線性回歸", h1="台北第二果菜批發市場<br>高麗菜均價",
        lead="2024 到 2026 年，台北市第二果菜批發市場每天都有高麗菜批發交易。請估計每一天的平均批發價。",
        target="平均批發價（元/公斤）", filter="月份",
        facts=[("796 天", "每一列是一天的行情"), ("8 個欄位", "價格、交易量、天氣")],
        row_note="一列是一天的行情。",
        cols=[
            ("日期", "交易日。"),
            ("月份", "交易日的月份，1 到 12。"),
            ("平均批發價（元/公斤）", "當天所有高麗菜交易的平均價格。"),
            ("高價位均價（元/公斤）", "當天價格最高的那一批交易的平均價，通常是品質最好的菜。"),
            ("低價位均價（元/公斤）", "當天價格最低的那一批交易的平均價，通常是品質較差的菜。"),
            ("交易量（噸）", "當天這個市場賣出的高麗菜總重量。"),
            ("台北近30日均溫（℃）", "到這一天為止，台北最近 30 天的平均氣溫。"),
            ("台北近7日雨量（毫米）", "到這一天為止，台北最近 7 天的雨量。"),
        ],
        notes="""    <div class="warn">
      <strong>想一想：</strong>估價的時候，當天的成交行情還不知道。只能用估價當下就拿得到的資訊當特徵。
    </div>""",
        data_note="天氣有空白，前處理時要決定怎麼辦。",
        steps=[
            ("資料", "讀進資料，看筆數、欄位、哪裡有空白。"),
            ("前處理", "決定用哪些欄，處理空白，必要時縮放。"),
            ("訓練", "訓練線性回歸，看係數，畫出貼合程度。"),
            ("你的判斷", "用自己的話寫結論，不要請 AI 寫。"),
        ],
        libs="numpy、matplotlib、scikit-learn",
        checks=[
            "這次用了哪些欄，沒用的欄為什麼不用，前處理留下多少列",
            "有沒有因尺度不同而縮放",
            "訓練後的係數",
            "一張看得出貼合程度的圖",
            "你自己的判斷：交易量變多時價格怎麼走，溫度或雨量有沒有改這個判斷",
        ],
    ),
    dict(
        dir=LOG, csv="mlb_games.csv", notebook="例行賽贏球.ipynb", title="例行賽贏球委託",
        svg=BALL_SVG, eyebrow="邏輯回歸", h1="美國職棒大聯盟<br>例行賽贏球",
        lead="2024 到 2025 年美國職棒大聯盟（MLB）的例行賽。例行賽是季後賽之前的正規賽程。請估計這一隊這場會不會贏。資料來源是 MLB Stats API，僅供課堂使用。",
        target="贏球（1為贏）", filter="球隊",
        facts=[("9718 列", "4859 場比賽，每場兩隊各一列"), ("12 個欄位", "比賽、打擊、守備、輸贏")],
        row_note="一列是一隊在一場比賽的紀錄。",
        cols=[
            ("球季", "哪一年的球季。"),
            ("日期", "比賽日期。"),
            ("比賽編號", "大聯盟給每場比賽的代號，同一場的兩列編號相同。"),
            ("球隊", "這一列是哪一隊（英文隊名）。"),
            ("主場（1為主場）", "1 表示這場在自己的球場打，0 表示到對手的球場打。"),
            ("得分（分）", "這一隊這場得了幾分。"),
            ("安打（支）", "這一隊的打者把球打進場內、自己安全上壘的次數。"),
            ("失誤（次）", "這一隊守備時出錯（漏接、傳歪等）的次數，失誤會讓對手多上壘。"),
            ("對手安打（支）", "對手這場的安打數。"),
            ("對手失誤（次）", "對手這場的失誤數。"),
            ("安打差（支）", "安打減掉對手安打，正數表示這一隊打得比較多。"),
            ("贏球（1為贏）", "1 表示這一隊贏，0 表示輸。"),
        ],
        notes="""    <p class="rule">每一場有兩列，兩隊各一列。同一場的安打差一正一負，贏球也一勝一負。這是賽後的解釋：安打比較多的一方為什麼比較會贏，不是賽前預測。</p>
    <div class="warn">
      <strong>想一想：</strong>賽後的資料都看得到，但有些欄本身就決定了輸贏。放進模型等於把答案放進去，這種欄不能當特徵。
    </div>""",
        data_note="選一隊來看，可以看到這隊整季的比賽。",
        steps=[
            ("資料", "讀進資料，看筆數、欄位，同一場是不是有兩列。"),
            ("前處理", "處理每場兩列、互為鏡像的問題，決定用哪些欄，必要時縮放。"),
            ("訓練", "訓練邏輯回歸，看係數，畫出贏球機率怎麼走。"),
            ("你的判斷", "用自己的話寫結論，不要請 AI 寫。"),
        ],
        libs="numpy、matplotlib、scikit-learn",
        checks=[
            "這次用了哪些欄，沒用的欄為什麼不用，前處理留下多少列",
            "有沒有因尺度不同而縮放",
            "訓練後的係數",
            "一張看得出贏球機率怎麼走的圖",
            "你自己的判斷：安打差變大時贏球機率怎麼走，主場或失誤有沒有改這個判斷",
        ],
    ),
    dict(
        dir=NN, csv="body_performance.csv", notebook="體能等級.ipynb", title="體能等級委託",
        svg=FIT_SVG, eyebrow="神經網路", h1="成人體能等級",
        lead="南韓國民體力檢測，20 到 64 歲的成人，約 2021 年以前測的。請估計這個人的體能等級。資料來源是韓國體育振興公團（南韓負責推廣國民運動的公家機構），經 Kaggle 使用者整理，CC0（公開授權，可自由使用）。",
        target="體能等級（A最好、D最差）", filter="體能等級（A最好、D最差）",
        facts=[("13393 人", "每一列是一個人的檢測結果"), ("12 個欄位", "身體、血壓、體能測驗、等級")],
        row_note="一列是一個人的檢測結果。",
        cols=[
            ("年齡（歲）", "受測時的年齡。"),
            ("性別（M為男、F為女）", "M 是男性，F 是女性。"),
            ("身高（公分）", "身高。"),
            ("體重（公斤）", "體重。"),
            ("體脂率（%）", "身體裡脂肪佔體重的百分比。"),
            ("舒張壓（mmHg）", "血壓的下面那個數字：心臟放鬆時，血液壓在血管上的力道。"),
            ("收縮壓（mmHg）", "血壓的上面那個數字：心臟用力把血送出去時，血液壓在血管上的力道。"),
            ("握力（公斤）", "用手握緊握力計時，最大能出多少力。"),
            ("坐姿體前彎（公分）", "坐在地上、雙腳伸直，身體往前彎、手往前伸的距離，看柔軟度。"),
            ("仰臥起坐（次）", "躺下再坐起來，做了幾下，看腹部肌耐力。"),
            ("立定跳遠（公分）", "站著不助跑，雙腳一起往前跳的距離，看腿部爆發力。"),
            ("體能等級（A最好、D最差）", "綜合上面的檢測給的等級。"),
        ],
        notes="""    <p class="rule">這次答案有 A、B、C、D 四類，不是兩類。四級的人數差不多。留一部分資料當測試，不要拿來訓練。正確率看測試資料。</p>
    <div class="warn">
      <strong>注意：</strong>資料沒有清理過，有些數值不合理。先看過每一欄，再決定怎麼處理。
    </div>""",
        data_note="",
        steps=[
            ("資料", "讀進資料，看筆數、欄位、各等級有幾人。"),
            ("前處理", "處理不合理的數值，轉成數字，分出訓練與測試資料，必要時縮放。"),
            ("訓練", "訓練神經網路，再跟沒有隱藏層的模型比正確率，畫出年齡怎麼改變等級。"),
            ("你的判斷", "用自己的話寫結論，不要請 AI 寫。"),
        ],
        libs="numpy、matplotlib、tensorflow",
        checks=[
            "這次用了哪些欄，沒用的欄為什麼不用，不合理的數值怎麼處理，前處理留下多少列",
            "有沒有因尺度不同而縮放",
            "網路有幾層，每層幾個單元",
            "測試資料的正確率，以及同一份資料只用輸出層、沒有隱藏層時的正確率",
            "一張看得出年齡怎麼改變等級的圖",
            "你自己的判斷：同樣的成績，年齡或性別不同時等級怎麼走；加上隱藏層差多少",
        ],
    ),
]


for p in PAGES:
    out = p["dir"] / "委託.html"
    out.write_text(page(p), encoding="utf-8")
    print(f"{out.relative_to(ROOT)}  {out.stat().st_size:,} bytes")
