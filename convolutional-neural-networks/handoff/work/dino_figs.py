import sys, os, numpy as np, torch, torch.nn.functional as Fn, torchvision.transforms.functional as TF
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as P
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False

src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
img = Image.open(src).convert('RGB'); W, H = img.size
torch.manual_seed(0)
trained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
untrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=False).eval()
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def prep(im, size): return TF.normalize(TF.to_tensor(im.resize(size)), MEAN, STD)[None]
@torch.no_grad()
def cls(model, im, s): return Fn.normalize(model.forward_features(prep(im, (s, s)))['x_norm_clstoken'][0], dim=0)
@torch.no_grad()
def patches(model, im, size):
    f = model.forward_features(prep(im, size))['x_norm_patchtokens'][0]
    return f, (size[1] // 14, size[0] // 14)

RED, CY, YEL = '#ff4d6d', '#36cfc9', '#ffd666'
cols = ['#ff7a45', '#36cfc9', '#ffd666', '#9254de', '#73d13d', '#ff4d6d']
def new(title, n=1, ratios=None, h=7.2):
    fig, axs = plt.subplots(1, n, figsize=(12.8, h), gridspec_kw=dict(width_ratios=ratios) if ratios else None)
    fig.suptitle(title, fontsize=23, y=.97); return fig, (list(axs) if n > 1 else [axs])
def show(ax, im, sub=None, ec=None):
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(ec or 'white'); s.set_linewidth(6 if ec else 0)
    if sub: ax.set_title(sub, fontsize=16)
def rect(ax, b, c, text=None, lw=4, ls='-'):
    ax.add_patch(P.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=lw, ls=ls))
    if text: ax.text(b[0] + 4, b[1] + 4, text, fontsize=15, va='top', color='black', bbox=dict(fc=c, ec='none', pad=2))
def save(fig, n, **kw):
    fig.subplots_adjust(top=kw.get('top', .87), bottom=.04, left=.03, right=.97, wspace=kw.get('ws', .12))
    fig.savefig(os.path.join(out, n), dpi=100, facecolor='white'); plt.close(fig); print('saved', n)

G1, G2 = (0, 40, 400, 440), (240, 0, 640, 400)
LOC = [('左貓的頭', (210, 50, 310, 150)), ('右貓的頭', (350, 260, 470, 380)), ('左貓的肚子', (40, 180, 160, 300)),
       ('遙控器', (50, 40, 170, 160)), ('沙發布', (480, 370, 600, 470))]

# q01
fig, a = new('DINOv2 的訓練資料：只有圖片，沒有任何標籤'); show(a[0], img)
a[0].text(W / 2, H - 25, '沒有人告訴它「這是貓」', fontsize=22, ha='center', color='white', bbox=dict(fc='black', alpha=.75))
save(fig, 'q01_nolabel.png')

# q02 global crops
fig, a = new('第 1 步：裁 2 個大塊（全域視角），各縮放成 224×224', 3, [1.5, 1, 1])
show(a[0], img); rect(a[0], G1, cols[0], '大塊 A'); rect(a[0], G2, cols[1], '大塊 B', ls='--')
show(a[1], img.crop(G1).resize((224, 224)), '大塊 A', cols[0]); show(a[2], img.crop(G2).resize((224, 224)), '大塊 B', cols[1])
save(fig, 'q02_global.png')

# q03 local crops
fig = plt.figure(figsize=(12.8, 7.2)); fig.suptitle('第 2 步：再裁幾個小塊（局部視角），縮放成 98×98', fontsize=23, y=.97)
ax = fig.add_axes([.02, .05, .55, .82]); show(ax, img)
for k, (nm, b) in enumerate(LOC): rect(ax, b, cols[k], nm)
for k, (nm, b) in enumerate(LOC):
    sa = fig.add_axes([.6 + (k % 2) * .2, .62 - (k // 2) * .28, .17, .24]); show(sa, img.crop(b).resize((98, 98)), nm, cols[k])
fig.savefig(os.path.join(out, 'q03_local.png'), dpi=100, facecolor='white'); plt.close(fig); print('saved q03')

# q04 masked
gA = np.array(img.crop(G1).resize((224, 224))).copy(); rng = np.random.default_rng(1)
mask = rng.random((16, 16)) < .4
for i in range(16):
    for j in range(16):
        if mask[i, j]: gA[i * 14:(i + 1) * 14, j * 14:(j + 1) * 14] = 128
fig, a = new('第 3 步：給學生的版本，還會遮掉一些 patch（示意）', 2)
show(a[0], img.crop(G1).resize((224, 224)), '老師看到：完整的大塊 A', cols[0])
show(a[1], gA, f'學生看到：遮掉 {int(mask.sum())} / 256 個 patch', RED)
save(fig, 'q04_mask.png')

# q05/q06 CLS similarity, untrained vs trained
def sim_fig(model, name, title):
    t = cls(model, img.crop(G1), 224)
    sims = [float(t @ cls(model, img.crop(b), 98)) for _, b in LOC]
    fig = plt.figure(figsize=(12.8, 7.2)); fig.suptitle(title, fontsize=22, y=.97)
    ta = fig.add_axes([.02, .3, .2, .4]); show(ta, img.crop(G1).resize((224, 224)), '老師：大塊 A', cols[0])
    ba = fig.add_axes([.33, .1, .64, .75])
    y = np.arange(len(LOC))[::-1]
    ba.barh(y, sims, color=[cols[k] for k in range(len(LOC))], height=.6)
    for k, s in enumerate(sims): ba.text(s + .01, y[k], f'{s:.2f}', va='center', fontsize=16)
    ba.set_yticks(y, [nm for nm, _ in LOC], fontsize=17); ba.set_xlim(0, 1.1); ba.set_xlabel('和老師輸出的相似度（1 = 完全一樣）', fontsize=15)
    ba.tick_params(axis='x', labelsize=13); [ba.spines[s].set_visible(False) for s in ['top', 'right']]
    fig.savefig(os.path.join(out, name), dpi=100, facecolor='white'); plt.close(fig)
    print(name, [(nm, round(s, 2)) for (nm, _), s in zip(LOC, sims)])
sim_fig(untrained, 'q05_sim_untrained.png', '訓練前（隨機權重）：數字亂跳，沒有規律')
sim_fig(trained, 'q06_sim_trained.png', '訓練後（真正的 DINOv2）：只有沙發布和大塊 A 不像')

# q07/q08 patch similarity maps
S = (644, 476)
def heat(model, pt):
    f, (gh, gw) = patches(model, img, S); f = Fn.normalize(f, dim=1)
    r, c = int(pt[1] / H * gh), int(pt[0] / W * gw)
    s = (f @ f[r * gw + c]).reshape(gh, gw).numpy()
    return s
def overlay(ax, s, pt, sub):
    ax.imshow(img); ax.imshow(Image.fromarray(np.uint8(plt.cm.inferno((s - s.min()) / (s.max() - s.min())) * 255)).resize((W, H), Image.NEAREST), alpha=.6)
    ax.scatter([pt[0]], [pt[1]], marker='*', s=700, c=CY, ec='black', zorder=3); ax.axis('off'); ax.set_title(sub, fontsize=17)
ear = (290, 100); couch = (540, 430)
fig, a = new('選左貓頭上一個 patch（★），看其他 patch 和它有多像（越亮越像）', 2)
overlay(a[0], heat(untrained, ear), ear, '訓練前：亂成一片，看不出貓')
overlay(a[1], heat(trained, ear), ear, '訓練後：兩隻貓的身體都亮起來')
save(fig, 'q07_heat_cat.png')
fig, a = new('換成沙發布上的 patch（★）', 2)
overlay(a[0], heat(untrained, couch), couch, '訓練前：只看顏色，淺色的毛也亮')
overlay(a[1], heat(trained, couch), couch, '訓練後：亮的是沙發布，貓是暗的')
save(fig, 'q08_heat_couch.png')

# q09 PCA
f, (gh, gw) = patches(trained, img, S); f = f.numpy(); f = f - f.mean(0)
u, s_, vt = np.linalg.svd(f, full_matrices=False); p = f @ vt[:3].T
p = (p - p.min(0)) / (p.max(0) - p.min(0))
fg = p[:, 0] > np.median(p[:, 0])
if fg.reshape(gh, gw)[0, :].mean() > .5: p[:, 0] = 1 - p[:, 0]
pca = Image.fromarray(np.uint8(p.reshape(gh, gw, 3) * 255)).resize((W, H), Image.NEAREST)
fig, a = new('DINOv2 每個 patch 的特徵壓成顏色：顏色像 = 特徵像', 2)
show(a[0], img, '原圖'); show(a[1], pca, '特徵圖：兩隻貓同一種粉紅，沙發是藍色')
save(fig, 'q09_pca.png')
