# 邏輯回歸實驗

把實驗夾寫進學生專案的 `邏輯回歸/`，並在該專案根目錄裝上 numpy、matplotlib、scikit-learn、ipykernel。

實驗夾裡有實驗本 `邏輯回歸.ipynb`、委託 `brief.md`、工作本 `mlb_wins.ipynb`，以及 `data/` 的三份 csv。實驗本做完後從 `brief.md` 開始。

已有 `邏輯回歸/` 時不覆寫檔案，只補還沒有的相依。不會改到已經放在同一專案裡的 `線性回歸/`。

```powershell
uvx --from git+https://github.com/mz038197/Machine-Learning-Lab.git@main#subdirectory=logistic-regression add-logistic-regression
```

本機：

```powershell
uvx --from . add-logistic-regression -C <學生專案目錄>
```
