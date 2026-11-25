# 線性回歸實驗

把實驗夾寫進學生專案的 `線性回歸/`，並在該專案根目錄裝上 numpy、matplotlib、scikit-learn、ipykernel、ipywidgets。

實驗夾裡有實驗本 `線性回歸.ipynb`、委託 `委託.html`、工作本 `高麗菜估價.ipynb`，以及 `data/` 的三份 csv。實驗本做完後，用瀏覽器打開 `委託.html` 開始。

已有 `線性回歸/` 時不覆寫檔案，只補還沒有的相依。

## 備案委託

`備案/尖峰負載/` 是高麗菜估價的備案，不在套件裡，安裝時學生不會拿到。要用時：

1. 把 `備案/尖峰負載/` 整個資料夾發給學生，例如壓成 zip。
2. 學生把 `尖峰負載/` 放進自己的 `線性回歸/` 實驗夾。
3. 學生用瀏覽器打開 `尖峰負載/委託.html` 開始。

相依和線性回歸一樣，不用再裝。老師版是 `teacher/尖峰負載_老師版.ipynb`。另開一張成績表給這份委託。

```powershell
uvx --from git+https://github.com/mz038197/Machine-Learning-Lab.git@main#subdirectory=linear-regression add-linear-regression
```

本機：

```powershell
uvx --from . add-linear-regression -C <學生專案目錄>
```
