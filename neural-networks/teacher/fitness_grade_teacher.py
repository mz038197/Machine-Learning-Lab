# 體能等級 參考解（老師對照用，不發給學生）
# 對應 fitness_grade.ipynb 的三格：1 資料、2 前處理、3 訓練。
# 只用 numpy、matplotlib、tensorflow。在學生專案的 神經網路/ 裡執行。

# %% 1 資料
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

raw = np.genfromtxt("data/body_performance.csv", delimiter=",", dtype=str, encoding="utf-8")
header, rows = raw[0], raw[1:]
print("筆數", len(rows))
print("欄位", list(header))

grades, counts = np.unique(rows[:, -1], return_counts=True)
print("各等級人數", dict(zip(grades, counts)))

# 性別 M→1、F→0；其餘欄位轉成數字
sex = (rows[:, 1] == "M").astype(float)
nums = rows[:, [0, 2, 3, 4, 5, 6, 7, 8, 9, 10]].astype(float)
X_all = np.column_stack([nums[:, 0], sex, nums[:, 1:]])
names = ["年齡", "性別", "身高", "體重", "體脂率", "舒張壓", "收縮壓", "握力", "坐姿體前彎", "仰臥起坐", "立定跳遠"]

for name, col in zip(names, X_all.T):
    print(f"{name:6s} 最小 {col.min():7.1f}  最大 {col.max():7.1f}")

# %% 2 前處理
# 一種清理方式：刪掉不可能的值。仰臥起坐 0 次可能是真的做不起來，這裡保留。
AGE, SEX, HEIGHT, WEIGHT, FAT, DIA, SYS, GRIP, BEND, SITUP, JUMP = range(11)
ok = (
    (X_all[:, DIA] >= 40)
    & (X_all[:, SYS] >= 70)
    & (X_all[:, GRIP] > 0)
    & (X_all[:, JUMP] > 0)
    & (X_all[:, BEND] > -20) & (X_all[:, BEND] < 50)
    & (X_all[:, FAT] >= 5) & (X_all[:, FAT] <= 50)
)
print("處理前", len(X_all), "列，處理後", ok.sum(), "列")

X = X_all[ok]
y = np.searchsorted(np.array(["A", "B", "C", "D"]), rows[ok, -1])  # A→0 … D→3

# 打亂後 80% 訓練、20% 測試
rng = np.random.default_rng(0)
order = rng.permutation(len(X))
cut = int(len(X) * 0.8)
train, test = order[:cut], order[cut:]

# 各欄尺度差很多（血壓上百、性別 0/1），用訓練資料的平均與標準差做標準化
mu = X[train].mean(axis=0)
sd = X[train].std(axis=0)
Xn = (X - mu) / sd
X_train, y_train = Xn[train], y[train]
X_test, y_test = Xn[test], y[test]
print("訓練", X_train.shape, "測試", X_test.shape)

# %% 3 訓練
def build(hidden):
    tf.random.set_seed(0)
    model = tf.keras.Sequential(
        [tf.keras.Input(shape=(X.shape[1],))]
        + [tf.keras.layers.Dense(units, activation="relu") for units in hidden]
        + [tf.keras.layers.Dense(4, activation="softmax")]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# 神經網路：兩層隱藏層 25、15，輸出層 4 個單元用 softmax
model = build((25, 15))
model.fit(X_train, y_train, epochs=80, batch_size=64, verbose=0)
_, acc = model.evaluate(X_test, y_test, verbose=0)
print("每層單元數", [layer.units for layer in model.layers], "測試正確率", round(acc, 3))

# 只有輸出層、沒有隱藏層（等於多類別的邏輯回歸）
flat = build(())
flat.fit(X_train, y_train, epochs=80, batch_size=64, verbose=0)
_, flat_acc = flat.evaluate(X_test, y_test, verbose=0)
print("沒有隱藏層 測試正確率", round(flat_acc, 3))

# 固定一組成績（各欄中位數），年齡 20→64，男女各畫一張
base = np.median(X, axis=0)
ages = np.arange(20, 65)
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, s, title in zip(axes, [1.0, 0.0], ["Male", "Female"]):
    grid = np.tile(base, (len(ages), 1))
    grid[:, AGE] = ages
    grid[:, SEX] = s
    prob = model.predict((grid - mu) / sd, verbose=0)
    for k, g in enumerate("ABCD"):
        ax.plot(ages, prob[:, k], label=g)
    ax.set_title(title)
    ax.set_xlabel("age")
axes[0].set_ylabel("probability")
axes[0].legend()
plt.show()
