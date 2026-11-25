# 交接：Transformer 逐步拆解互動網頁

日期：2026-11-25

## 這是什麼

使用者是老師，要一個給學生看的互動網頁，把 GPT 式 Transformer 從輸入字串到輸出結果的每一步，用可視化方式一步步拆開。全部回覆用**繁體中文**。

- 已發布的網頁（私人 Artifact，目前第 31 版）：https://claude.ai/artifact/NxXe3HYnBKCjzVo6tGtd31
- 網頁**不在 repo 裡**，原始檔都在上一個 session 的暫存資料夾（見下方）。之前問過要不要存進 repo，使用者沒回答。

## 檔案位置（暫存資料夾，重開機可能被清掉）

`C:\Users\mz038\AppData\Local\Temp\claude\eb130486cdaa1318\1aed3183-fcb4-493c-8190-e22c9646b583\scratchpad\`

| 檔案 | 用途 |
|---|---|
| `page.src.html` | **唯一要改的原始碼**。模型資料位置是 `const MODEL=/*MODEL*/null;` |
| `model.json`、`fixtures.json` | `train.py` 產生的模型權重與 PyTorch 對照答案 |
| `train.py` | 訓練小模型：`uv run --no-project --python 3.12 --with torch --with numpy python train.py 7` |
| `build_check.js` | `node build_check.js`：把模型塞進原始碼，輸出 `transformer.html`，並在 vm 裡跑全部檢查 |
| `transformer.html` | 建置結果，發布的就是這個檔 |
| `page.src.before-*.html` | 每次修改前的備份 |
| `patch_*.js`、`stage_*.txt` | 歷次修改腳本與替換片段（已套用，只供參考） |

**如果暫存資料夾不見了：** 用 Artifact 工具 `action: "read"` 讀上面的網址，拿回完整 HTML。把其中 `const MODEL={...};` 換回 `const MODEL=/*MODEL*/null;`，就是 `page.src.html`。`build_check.js` 和 `train.py` 則需要重寫。

## 工作流程

1. 用 node 腳本，以「必須剛好出現一次」的字串替換修改 `page.src.html`（檔案有超長行，不適合用 Edit）。
2. `node build_check.js`，必須全部通過。目前輸出包括：
   - PyTorch 差 4.93e-7
   - 65 個步驟畫面
   - bpe ok
   - walk
   - layer diagrams 84 views
   - layer zoom 216 views
   - matrix stage
3. 用 headless Chrome 截圖檢查版面（`C:\Program Files\Google\Chrome\Application\chrome.exe --headless=new --screenshot=... --window-size=1280,3600`）。
   - 預覽檔的做法：在 `transformer.html` 尾端加一段 `<script>`，設定 `S` 狀態後呼叫 `go(...)`。
   - 手機版：用 400px 寬的 iframe 包起來截圖，因為 Chrome 最小寬度是 500px。
4. 用 Artifact 工具發布同一個 `transformer.html`，帶 `url` 參數；第一次在新 session 發布前要先 `read`。

## 模型與頁面現況

- 模型：
  - 規模：`D=24, HEADS=2, HD=12, LAYERS=2, FF=48`，只用 30 句中文訓練。
  - 字典與切詞：41 個 token，單詞級最長比對。
  - 結構：Pre-LN、ReLU、sin 位置編碼。
- 全頁維度統一用**列向量、矩陣乘在右邊**：`q = (e + p) W_q`、`X(n×24)·W_q₁(24×12)`。
- 現在共 12 步（00～12）：全貌、Tokenization、Token ID、Embedding、Positional Encoding、相加、**STEP 06 Q/K/V**、Multi-Head、堆疊多層、Linear→Logits、Softmax→機率、選字、訓練。
- **STEP 01**：
  - 開場只顯示切詞結果，不說用的是哪種切法。
  - 接著依序示範 A 單詞級、B 字元級（都可一塊一塊切）、C BPE（英文 low/lower/newest/widest，一步步合併）。
  - 最後是「三種切法比一比」與優缺點表。
- **STEP 06 第 3 段「層和頭是什麼」是核心**，層層點開：
  1. 層圖裡六個部分都能點（`S.lz`：`ln1` / `attn` / `res1` / `ln2` / `ffn` / `res2`），點了下面就換成那一塊的放大流程圖（共用 `flowSvg`，展開哪個方塊由 `S.lbox` 記錄）。
  2. `attn` 會打開「頭在哪裡」圖，點 Scaled Dot-Product 方塊再打開五個方塊的流程圖（`S.sbox`）。
  3. Mask 展開時，裡面有遮罩前後的數字大表（用 foreignObject）。
- 原本獨立的 Score、Mask、Softmax、加權 V、殘差、前饋網路六步，都已併進上面的流程圖並刪除。舊的架構圖連結透過別名導到對應的放大圖（`score` / `mask` / `softmax` / `weighted` / `z-ln1` 等）。

## 使用者偏好（重要）

- 使用者說「先不要動作」或「重述一次」時，只重述理解並確認，不要改檔。
- 喜歡**一層一層點開、在對應方塊旁邊展開**，不喜歡把內容獨立成卡片或另放在圖的下方。
- 比喻：Q＝外派者的問題卡、K＝海報標題、V＝海報內容。
- 說明要具體、有真實數字，不要抽象。

## 還沒做或沒驗證的

- **點擊從沒在真實瀏覽器試過**（展開方塊、點格子換列、層圖切換、架構圖跳轉、鍵盤操作），只在 vm 裡渲染過。
- **手機版**：STEP 06 的流程圖與 STEP 01 的新內容都沒有重新看過。流程圖手機上要左右捲動。
- 16 個 token 的長句，Mask 大表縮小後沒有再截圖確認。
- Mask 大表會顯示「-0.0」，之前問過要不要改成「0.0」，使用者沒回答。
- 之前提過但使用者沒回答的兩個提議：
  - STEP 07 Multi-Head 加「移除 W_o」的開關。
  - 把網頁存進 repo。
- Artifact 是私人的，要分享給學生得由使用者從 Share 選單開放。

## 建議使用的 skills

- `artifact-design`：修改或重新發布網頁前載入，裡面有頁面規範。
- `claude-in-chrome`：在真實 Chrome 裡測試點擊與手機寬度，這是目前最大的缺口。
- `dataviz`：如果要新增圖表或調整配色。
