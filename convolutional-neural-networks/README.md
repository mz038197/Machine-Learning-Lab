# 卷積神經網路實驗

把實驗夾寫進學生專案的 `卷積神經網路/`，並在該專案根目錄裝上 numpy、matplotlib、tensorflow、ipykernel。

實驗夾裡有實驗本 `卷積神經網路.ipynb`，以及 `data/mnist_small.npz`：從 MNIST 取出的 6000 張訓練、1000 張測試手寫數字，CC BY-SA 3.0。這份實驗還沒有委託。

已有 `卷積神經網路/` 時不覆寫檔案，只補還沒有的相依。不會改到已經放在同一專案裡的 `神經網路/` 或其他實驗夾。

```powershell
uvx --from git+https://github.com/mz038197/Machine-Learning-Lab.git@main#subdirectory=convolutional-neural-networks add-convolutional-neural-networks
```

本機：

```powershell
uvx --from . add-convolutional-neural-networks -C <學生專案目錄>
```
