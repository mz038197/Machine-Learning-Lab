# 線性回歸實驗

把一份線性回歸實驗放進學生自己的專案，並讓那個專案帶著跑這份實驗所需要的相依。

## Language

**實驗夾**:
學生專案裡名為「線性回歸」的資料夾。裡面是這份實驗的 notebook 和它的資料。
_Avoid_: 把 notebook 和資料散在專案根目錄, lab, linear-regression

**學生專案**:
學生執行指令時所在的那個目錄。不往上層另找專案。實驗夾在這裡面，相依也屬於這裡的根目錄環境。
_Avoid_: Machine-Learning-Lab, 模板, 上層專案

**相依**:
這份線性回歸實驗要跑起來所需要的 Python 套件，屬於學生專案根目錄那一個環境。實驗夾裡沒有另一套環境。
_Avoid_: 只裝在老師的 Machine-Learning-Lab, 實驗夾自己的環境

**實驗本**:
這份線性回歸實驗唯一的 notebook。住在本專案的 `linear-regression` 目錄裡。
_Avoid_: 在 lab/teacher 再留一本, 執行時才下載的另一本, linear-regression-tool

**實驗資料**:
跟著實驗本、要放進實驗夾的 ex1data1.txt 與 ex1data2.txt。本專案根目錄 data/ 裡的同名檔仍給舊講義用。兩邊各算各的，改一邊不會更新另一邊。
_Avoid_: 只留根目錄那一份, 執行時再下載, 兩邊自動同步
