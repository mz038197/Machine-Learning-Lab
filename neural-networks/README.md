# 神經網路實驗

把實驗夾寫進學生專案的 `神經網路/`，並在該專案根目錄裝上 numpy、matplotlib、tensorflow、ipykernel。

實驗夾裡只有 notebook。咖啡烘焙的資料在 notebook 裡造出來。

已有 `神經網路/` 時不覆寫檔案，只補還沒有的相依。不會改到已經放在同一專案裡的 `線性回歸/` 或 `邏輯回歸/`。

```powershell
uvx --from git+https://github.com/mz038197/Machine-Learning-Lab.git@main#subdirectory=neural-networks add-neural-networks
```

本機：

```powershell
uvx --from . add-neural-networks -C <學生專案目錄>
```
