import json
from pathlib import Path


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [line if line.endswith("\n") else line + "\n" for line in text.splitlines()]}


def code(src: str, *, editable: bool | None = None, deletable: bool | None = None, exec_count=None) -> dict:
    meta = {}
    if editable is not None:
        meta["editable"] = editable
    if deletable is not None:
        meta["deletable"] = deletable
    return {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": meta,
        "outputs": [],
        "source": [line if line.endswith("\n") else line + "\n" for line in src.splitlines()],
    }


def main() -> None:
    nb_path = Path("lab/teacher/Regression/11_Multiple_Linear_Regression.ipynb")
    nb_path.parent.mkdir(parents=True, exist_ok=True)

    setup_code = """# region 資料載入
import copy
import math
import sys, os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
%matplotlib inline

def find_repo_root(marker="README.md"):
    cur = Path.cwd()
    while cur != cur.parent:  # 防止無限迴圈，到達檔案系統根目錄就停
        if (cur / marker).exists():
            return cur
        cur = cur.parent
    return None

def import_data_from_github():
    import urllib.request

    def isRunningInColab() -> bool:
        return "google.colab" in sys.modules

    def isRunningInJupyterLab() -> bool:
        try:
            import jupyterlab  # noqa: F401
            return True
        except ImportError:
            return False

    def detect_env():
        from IPython import get_ipython
        if isRunningInColab():
            return "Colab"
        elif isRunningInJupyterLab():
            return "JupyterLab"
        elif "notebook" in str(type(get_ipython())).lower():
            return "Jupyter Notebook"
        else:
            return "Unknown"

    def get_utils_dir(env):
        if env == "Colab":
            if "/content" not in sys.path:
                sys.path.insert(0, "/content")
            return "/content/utils"
        else:
            return Path.cwd() / "utils"

    def get_data_dir(env):
        if env == "Colab":
            if "/content" not in sys.path:
                sys.path.insert(0, "/content")
            return "/content/data"
        else:
            return Path.cwd() / "data"

    env = detect_env()
    UTILS_DIR = get_utils_dir(env)
    DATA_DIR = get_data_dir(env)

    REPO_DIR = "Machine-Learning-Lab"
    os.makedirs(UTILS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    BASE = f"https://raw.githubusercontent.com/mz038197/{REPO_DIR}/main"

    utils_list = ["utils.py", "public_tests.py"]
    for u in utils_list:
        urllib.request.urlretrieve(f"{BASE}/utils/{u}", f"{UTILS_DIR}/{u}")

    data_list = ["ex1data1.txt", "ex1data2.txt"]
    for d in data_list:
        urllib.request.urlretrieve(f"{BASE}/data/{d}", f"{DATA_DIR}/{d}")

repo_root = find_repo_root()
if repo_root is None:
    import_data_from_github()
    repo_root = Path.cwd()

os.chdir(repo_root)
print(f"✅ 切換工作目錄至 {Path.cwd()}")
sys.path.append(str(repo_root)) if str(repo_root) not in sys.path else None
print("✅ 加入到系統路徑")

from utils.utils import *

plt.style.use("utils/deeplearning.mplstyle")
print("✅ 匯入模組及設定繪圖樣式")
# endregion 資料載入
"""

    nb = {
        "cells": [
            md(
                "## 練習測驗：多變量線性迴歸（Multiple Linear Regression）\n"
                "\n"
                "延續上一份「連鎖餐廳執行長」的情境：你想評估要在哪些城市開設新的分店。\n"
                "這次不只看人口數，還會同時考慮另一個重要因素（例如：當地平均收入）。\n"
                "\n"
                "## 大綱\n"
                "- 1 - 套件（Packages）\n"
                "- 2 - 多變量線性迴歸\n"
                "  - 2.1 問題說明（Problem Statement）\n"
                "  - 2.2 資料集（Dataset）\n"
                "  - 2.3 特徵縮放（Feature Scaling）\n"
                "  - 2.4 成本函數（Compute Cost）\n"
                "  - 2.5 梯度計算（Compute Gradient）\n"
                "  - 2.6 梯度下降（Gradient Descent）\n"
            ),
            md(
                "_**注意：** 為了避免自動評分系統（autograder）出錯，**請不要**編輯或刪除這份 notebook 中「非評分」的程式碼區塊，也請**不要新增**任何新的程式碼儲存格。_"
            ),
            md("## 1 - 套件（Packages）\n\n請先執行下方儲存格完成環境設定。"),
            code(setup_code, editable=False, deletable=False, exec_count=None),
            md(
                "## 2 - 多變量線性迴歸\n"
                "\n"
                "### 2.1 問題說明（Problem Statement）\n"
                "你是一間連鎖餐廳的執行長，正在評估要在哪些城市開設新的分店。\n"
                "\n"
                "與單變量不同的是：你希望同時利用多個特徵來預測獲利，例如：\n"
                "- 城市人口數（population）\n"
                "- 當地平均收入（income proxy）\n"
                "\n"
                "你的目標是用這些特徵，預測每個城市的餐廳平均每月獲利。"
            ),
            md("### 2.2 資料集（Dataset）\n\n載入多變量資料集："),
            code(
                "X_train, y_train = load_data_multi()\n"
                "print('Type of X_train:', type(X_train))\n"
                "print('Type of y_train:', type(y_train))\n"
                "print('First row of X_train:', X_train[0])\n"
                "print('First five elements of y_train:', y_train[:5])",
                editable=False,
                deletable=False,
            ),
            md(
                "#### 變數說明\n"
                "- `X_train`：shape 為 (m, n) 的特徵矩陣（m 筆資料、n 個特徵）\n"
                "  - 本測驗中 n = 2\n"
                "  - `X_train[:,0]`：城市人口（單位：10,000 人）\n"
                "  - `X_train[:,1]`：當地平均收入指標（單位：\\$10,000）\n"
                "- `y_train`：shape 為 (m,) 的標籤向量\n"
                "  - 每月獲利（單位：\\$10,000）"
            ),
            code(
                "print('The shape of X_train is:', X_train.shape)\n"
                "print('The shape of y_train is:', y_train.shape)\n"
                "print('Number of training examples (m):', X_train.shape[0])\n"
                "print('Number of features (n):', X_train.shape[1])",
                editable=False,
                deletable=False,
            ),
            md(
                "### 2.3 特徵縮放（Feature Scaling）\n"
                "多變量線性迴歸通常需要做特徵縮放（例如 z-score normalization），讓梯度下降更穩定、收斂更快。\n"
            ),
            code(
                "# UNQ_C1\n"
                "# GRADED FUNCTION: zscore_normalize_features\n"
                "def zscore_normalize_features(X):\n"
                "    \"\"\"Z-score normalize features.\n"
                "\n"
                "    Args:\n"
                "        X (ndarray): Shape (m, n) input features\n"
                "\n"
                "    Returns:\n"
                "        X_norm (ndarray): normalized features, shape (m, n)\n"
                "        mu (ndarray): mean of each feature, shape (n,)\n"
                "        sigma (ndarray): std of each feature, shape (n,)\n"
                "    \"\"\"\n"
                "    ### START CODE HERE ###\n"
                "    \n"
                "    ### END CODE HERE ###\n"
                "    return X_norm, mu, sigma\n",
                deletable=False,
            ),
            code(
                "X_norm, mu, sigma = zscore_normalize_features(X_train)\n"
                "print('mu:', mu)\n"
                "print('sigma:', sigma)\n"
                "print('First row of X_norm:', X_norm[0])\n"
                "\n"
                "from utils.public_tests import *\n"
                "zscore_normalize_features_test(zscore_normalize_features)",
                editable=False,
                deletable=False,
            ),
            md(
                "### 2.4 成本函數（Compute Cost）\n"
                "多變量線性迴歸模型：\n"
                "$$f_{w,b}(\\mathbf{x}) = \\mathbf{x}\\cdot\\mathbf{w} + b$$\n"
                "\n"
                "成本函數：\n"
                "$$J(\\mathbf{w},b) = \\frac{1}{2m} \\sum_{i=0}^{m-1} (f_{w,b}(\\mathbf{x}^{(i)}) - y^{(i)})^2$$\n"
            ),
            code(
                "# UNQ_C2\n"
                "# GRADED FUNCTION: compute_cost_multi\n"
                "def compute_cost_multi(X, y, w, b):\n"
                "    \"\"\"Compute cost for multivariate linear regression.\n"
                "\n"
                "    Args:\n"
                "        X (ndarray): Shape (m, n) features\n"
                "        y (ndarray): Shape (m,) target values\n"
                "        w (ndarray): Shape (n,) model parameters\n"
                "        b (scalar): bias parameter\n"
                "    Returns:\n"
                "        cost (float): scalar cost value\n"
                "    \"\"\"\n"
                "    m = X.shape[0]\n"
                "    ### START CODE HERE ###\n"
                "    \n"
                "    ### END CODE HERE ###\n"
                "    return cost\n",
                deletable=False,
            ),
            code(
                "initial_w = np.zeros(X_norm.shape[1])\n"
                "initial_b = 0.\n"
                "cost = compute_cost_multi(X_norm, y_train, initial_w, initial_b)\n"
                "print(f'Cost at initial w,b (zeros): {cost:.3f}')\n"
                "\n"
                "from utils.public_tests import *\n"
                "compute_cost_multi_test(compute_cost_multi)",
                editable=False,
                deletable=False,
            ),
            md(
                "### 2.5 梯度計算（Compute Gradient）\n"
                "梯度：\n"
                "$$\\frac{\\partial J}{\\partial b} = \\frac{1}{m}\\sum_{i=0}^{m-1} (f_{w,b}(\\mathbf{x}^{(i)})-y^{(i)})$$\n"
                "$$\\frac{\\partial J}{\\partial \\mathbf{w}} = \\frac{1}{m} X^T(\\mathbf{f}-\\mathbf{y})$$\n"
            ),
            code(
                "# UNQ_C3\n"
                "# GRADED FUNCTION: compute_gradient_multi\n"
                "def compute_gradient_multi(X, y, w, b):\n"
                "    \"\"\"Compute gradient for multivariate linear regression.\n"
                "    \n"
                "    Args:\n"
                "        X (ndarray): Shape (m, n)\n"
                "        y (ndarray): Shape (m,)\n"
                "        w (ndarray): Shape (n,)\n"
                "        b (scalar)\n"
                "    Returns:\n"
                "        dj_dw (ndarray): Shape (n,)\n"
                "        dj_db (scalar)\n"
                "    \"\"\"\n"
                "    m = X.shape[0]\n"
                "    ### START CODE HERE ###\n"
                "    \n"
                "    ### END CODE HERE ###\n"
                "    return dj_dw, dj_db\n",
                deletable=False,
            ),
            code(
                "tmp_dj_dw, tmp_dj_db = compute_gradient_multi(X_norm, y_train, initial_w, initial_b)\n"
                "print('dj_dw at initial w,b (zeros):', tmp_dj_dw)\n"
                "print('dj_db at initial w,b (zeros):', tmp_dj_db)\n"
                "\n"
                "from utils.public_tests import *\n"
                "compute_gradient_multi_test(compute_gradient_multi)",
                editable=False,
                deletable=False,
            ),
            md(
                "### 2.6 梯度下降（Gradient Descent）\n"
                "以下提供一個批次梯度下降（batch gradient descent）的參考實作（非評分）。\n"
            ),
            code(
                "def gradient_descent_multi(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):\n"
                "    \"\"\"Batch gradient descent for multivariate linear regression.\"\"\"\n"
                "    m = X.shape[0]\n"
                "    w = copy.deepcopy(w_in)\n"
                "    b = b_in\n"
                "    J_history = []\n"
                "\n"
                "    for i in range(num_iters):\n"
                "        dj_dw, dj_db = gradient_function(X, y, w, b)\n"
                "        w = w - alpha * dj_dw\n"
                "        b = b - alpha * dj_db\n"
                "        if i < 100000:\n"
                "            J_history.append(cost_function(X, y, w, b))\n"
                "        if i % max(1, math.ceil(num_iters / 10)) == 0:\n"
                "            print(f\"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}\")\n"
                "    return w, b, J_history\n"
            ),
            code(
                "initial_w = np.zeros(X_norm.shape[1])\n"
                "initial_b = 0.\n"
                "alpha = 0.1\n"
                "iterations = 1000\n"
                "\n"
                "w, b, J_hist = gradient_descent_multi(X_norm, y_train, initial_w, initial_b,\n"
                "                                     compute_cost_multi, compute_gradient_multi,\n"
                "                                     alpha, iterations)\n"
                "print('w,b found by gradient descent:', w, b)",
                editable=False,
                deletable=False,
            ),
            md(
                "#### 進階：用學到的參數做預測\n"
                "試著輸入一個新城市的特徵（人口、平均收入），並用你學到的 `(w,b)` 預測獲利。\n"
                "提示：要先用同一組 `mu`、`sigma` 做 z-score normalization，再丟進模型。"
            ),
        ],
        "metadata": {
            "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.13.0",
                "mimetype": "text/x-python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote: {nb_path}")


if __name__ == '__main__':
    main()


