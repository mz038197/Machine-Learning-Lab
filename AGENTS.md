## Agent skills

### Issue tracker

Issues and specs live as GitHub issues on `mz038197/Machine-Learning-Lab`. See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

## 委託.html

三個實驗的 `委託.html` 由 `tools/build_briefs.py` 產生，委託文字也寫在那支程式裡。改委託內容、版面或委託用的 CSV 時，改程式再重跑 `uv run python tools/build_briefs.py`，產出的 HTML 不要手改。
