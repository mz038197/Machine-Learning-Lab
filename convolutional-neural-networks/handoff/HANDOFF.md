# 交接文件：RF-DETR／DINOv2 圖解教學（2026-11-05）

## 使用者與目標
- 使用者正在學 RF-DETR，用**繁體中文（zh-TW）**溝通。所有回覆和草稿都要用 zh-TW。
- 使用者要的是「一句話配一張圖」：每張都是真實模型輸出的圖，而且要和旁白逐句對上。只有文字的內容看不懂。不夠細也不行，曾被說過「圖片沒有對上內容，並且顆粒度不夠細」。
- 配音：使用者說「先不用」，**不要主動提**。如果之後要做：Windows 有 SAPI zh-TW 語音，也有 ffmpeg，但沒有設定 TTS。

## 已完成的成品（用 answer-me-with-html 產生）
| 成品 | 路徑 |
|---|---|
| RF-DETR 說明頁 | `C:/Users/mz038/.answer-me-with-html/pages/rf-detr-20261104-175443.html` |
| CNN → RF-DETR 技術演進頁 | `C:/Users/mz038/.answer-me-with-html/pages/cnn-rf-detr-20261105-001246.html` |
| RF-DETR 影片（22 個畫面，約 342 秒，已更正層號） | `C:/Users/mz038/.answer-me-with-html/videos/rf-detr-20261105-002632.html` |
| DINOv2 影片（28 個畫面，約 11.7 分鐘：逐層架構 → 老師學生遊戲 → 訓練前後對照 → RF-DETR 怎麼用） | `C:/Users/mz038/.answer-me-with-html/videos/dinov2-20261105-002633.html` |
| 最早的純文字影片（已過時，可以刪） | `C:/Users/mz038/.answer-me-with-html/videos/rf-detr-20261104-221934.html` |

## 本資料夾的內容
原本的檔案在 session 暫存區，系統可能會清掉，所以已全部複製過來。
- `work/figs2.py`：產生 RF-DETR 影片的圖，輸出到 `fig2/p01–p22`。用法是 `python figs2.py <img> <outdir>`。
- `work/dino_figs.py`：產生 DINOv2 訓練與結果的圖，輸出到 `fig3/q01–q09`。
- `work/arch_figs.py`：產生 DINOv2 逐層架構的圖，輸出到 `fig4/r01–r16` 和 `info.json`。它在 448×448 上手動一層一層前向計算，並用 `assert torch.allclose` 確認結果和模型本身一致。
- `work/figs.py`：第一版的圖，已被 figs2 取代。`probe*.py` 和 `conv.py` 是查模型結構用的小腳本。
- `work/rfdetr_video.md`、`work/dinov2_video.md`：影片的草稿，圖片路徑都改成相對路徑 `../figN/...`。
- `dl/cats.jpg`：COCO val2017 000000039769（兩隻貓、遙控器、沙發，640×480）。

## 環境
- 原本的 venv 在暫存區：`uv venv --system-site-packages` 加上 `uv pip install rfdetr scipy`。要重建就照這兩步再做一次。
- 執行時用 `python -I -X utf8`，並設 `PYTHONIOENCODING=utf-8`。
- DINOv2 用 `torch.hub.load('facebookresearch/dinov2','dinov2_vits14')` 載入。未訓練版本加 `pretrained=False`。
- 重新產生影片：
  `node "C:/Users/mz038/.claude/skills/answer-me-with-html/scripts/am.mjs" video <draft.md> --no-open`
  - 旁白一行以 `>` 開頭，算一拍。
  - 圖片寫成 `![說明](path)`。
  - 產生前要把 STE 警告修到 0。STE 會抓連續的「的」、「多次」、「以下」這類詞，以及過長的句子。

## 已確認的關鍵事實（不要再弄錯）
- **RF-DETR Base 的投影器拿的是第 2、5、8、11 層的輸出**（設定是 out_feature_indexes=[2,5,8,11]）。第 3、6、9、12 層是全域注意力，其餘 8 層是視窗注意力（num_windows=4）。我之前一度講錯成 3/6/9/12，已更正並告知使用者。
- RF-DETR Base 的設定：
  - backbone 是 dinov2_windowed_small，patch 14，解析度 560，300 個 query，group_detr=13。
  - decoder 有 3 層、16 個 head、每個 head 2 個取樣點、1 個層級，所以每個 query 取樣 32 個點。
  - 總參數 32,174,530，其中卷積只佔約 5%。
- 貓圖的偵測結果：分數大於 0.5 的有 5 個（貓 0.96 和 0.93、遙控器 0.90 和 0.71、沙發 0.71），300 個 query 中有 292 個分數小於 0.1。
- DINOv2 ViT-S/14：
  - 384 維，12 個 block，6 個 head × 64 維。
  - MLP 是 1536 維中間層加 GELU。
  - pos_embed 是 37×37 的網格。
- 「示意」畫面（不是真實模型輸出）：密集框、NMS 規則、遮 patch、訓練專用的頭、匈牙利匹配用的假標註。

## 可能的下一步（使用者還沒要求）
- 配音，要等使用者開口才做。
- 使用者所在的資料夾 `convolutional-neural-networks` 是 CNN 課程專案，下一個 session 可能要把這些內容接到 CNN 教材裡。開始前先看該專案的 `README.md`。

## 建議載入的 skills
- `answer-me-with-html`：產生頁面和影片都要用。改內容時先載入它，再用 `am video` 或 `am render`。
- `dataviz`：要新增 matplotlib 圖的時候用。
