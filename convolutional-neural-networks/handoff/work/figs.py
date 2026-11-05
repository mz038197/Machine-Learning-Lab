import sys, os, numpy as np, torch, torchvision.transforms.functional as TF
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as P
from scipy.optimize import linear_sum_assignment
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
from rfdetr import RFDETRBase

src, out = sys.argv[1], sys.argv[2]
img = Image.open(src).convert('RGB'); W, H = img.size
from rfdetr.assets.coco_classes import COCO_CLASSES
m = RFDETRBase(); net = m.model.model.eval(); names = COCO_CLASSES
R = 560
x = TF.normalize(TF.to_tensor(img.resize((R, R))), m.means, m.stds)[None]
cap = {}
ca = net.transformer.decoder.layers[-1].cross_attn

def pre(mod, args, kwargs):
    a = list(args) + [None] * 6
    cap['q'] = kwargs.get('query', a[0]); cap['ref'] = kwargs.get('reference_points', a[1])
ca.register_forward_pre_hook(pre, with_kwargs=True)

with torch.no_grad(): o = net(x)
logits, boxes = o['pred_logits'][0], o['pred_boxes'][0]
prob = logits.sigmoid(); score, cls = prob.max(-1)

def xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], -1)

bx = xyxy(boxes).numpy(); sc = score.numpy(); cl = cls.numpy()
order = np.argsort(-sc)

def name(c):
    try:
        return names[int(c)] if isinstance(names, (list, tuple)) else names.get(int(c), str(c))
    except Exception:
        return str(c)
zh = {'cat': '貓', 'remote': '遙控器', 'couch': '沙發', 'bed': '床', 'dog': '狗'}
def lab(c): n = name(c); return zh.get(n, n)

final = [i for i in order if sc[i] > 0.5]
print('final', [(name(cl[i]), round(float(sc[i]), 2)) for i in final])
print('score>0.5:', len(final), ' <0.1:', int((sc < 0.1).sum()), ' total', len(sc))
cols = ['#ff7a45', '#36cfc9', '#ffd666', '#9254de', '#73d13d', '#ff4d6d']

def base(ax, title=None):
    ax.imshow(img); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    if title: ax.set_title(title, fontsize=15)

def box(ax, b, c, lw=3, ls='-', alpha=1, text=None):
    ax.add_patch(P.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=lw, ls=ls, alpha=alpha))
    if text: ax.text(b[0] + 4, b[1] + 16, text, color='black', fontsize=12, bbox=dict(fc=c, ec='none', alpha=.9, pad=2))

def save(fig, n):
    fig.savefig(os.path.join(out, n), dpi=110, bbox_inches='tight', facecolor='white'); plt.close(fig)

# 1 photo
fig, ax = plt.subplots(figsize=(6.4, 4.8)); base(ax); save(fig, '01_photo.png')

# 2 dense boxes (illustrative) vs after NMS
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.9))
base(axs[0], '傳統偵測器：同一物件被框很多次（示意）'); base(axs[1], 'NMS 刪掉重疊框之後')
for k, i in enumerate(final):
    b = bx[i]; w, h = b[2] - b[0], b[3] - b[1]
    for _ in range(9):
        j = b + rng.normal(0, 1, 4) * np.array([w, h, w, h]) * 0.07
        box(axs[0], j, cols[k % 6], lw=1.6, alpha=.75)
    box(axs[1], b, cols[k % 6], text=lab(cl[i]))
save(fig, '02_nms.png')

# 3 patches and windows
P14 = 14; g = R // P14
fig, axs = plt.subplots(1, 2, figsize=(12.8, 6.4))
im560 = img.resize((R, R))
for ax in axs: ax.imshow(im560); ax.axis('off')
for t in range(0, R + 1, P14):
    for ax in axs:
        ax.axhline(t, color='white', lw=.4, alpha=.8); ax.axvline(t, color='white', lw=.4, alpha=.8)
axs[0].set_title(f'縮放成 {R}×{R}，切成 {g}×{g} = {g*g} 個 14×14 patch', fontsize=14)
ax = axs[1]; ax.set_title('視窗注意力：切成 4 個視窗，各自在內部互看', fontsize=14)
for k, (r0, c0) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    ax.add_patch(P.Rectangle((c0 * R / 2, r0 * R / 2), R / 2, R / 2, fill=True, fc=cols[k], alpha=.28, ec=cols[k], lw=4))
pr, pc = 12, 10
axs[0].add_patch(P.Rectangle((pc * P14, pr * P14), P14, P14, fill=False, ec='#ff4d6d', lw=3))
axs[0].annotate('1 個 patch = 1 個 token', xy=(pc * P14 + 7, pr * P14), xytext=(250, 40), color='white', fontsize=13,
                bbox=dict(fc='#ff4d6d', ec='none'), arrowprops=dict(arrowstyle='->', color='#ff4d6d', lw=2))
save(fig, '03_patches.png')

# 4 all 300 queries
fig, axs = plt.subplots(1, 2, figsize=(13, 4.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
base(axs[0], '300 個 query 各交一個框：灰色 = 分數低，等於「沒東西」')
for i in order[::-1]:
    if sc[i] < 0.5: box(axs[0], bx[i], '#bfbfbf', lw=.6, alpha=.35)
for k, i in enumerate(final): box(axs[0], bx[i], cols[k % 6], text=f'{lab(cl[i])} {sc[i]:.2f}')
ax = axs[1]; s = sc[order]
ax.bar(range(len(s)), s, color=['#ff7a45' if v > .5 else '#bfbfbf' for v in s], width=1.0)
ax.axhline(.5, color='#ff4d6d', ls='--'); ax.text(len(s) * .35, .53, '門檻 0.5', color='#ff4d6d', fontsize=12)
ax.set_title(f'300 個 query 的信心分數（由高到低）\n只有 {len(final)} 個超過門檻', fontsize=14)
ax.set_xlabel('query 排名'); ax.set_ylabel('信心分數'); ax.set_ylim(0, 1)
save(fig, '04_queries.png')

# 5 deformable attention: real sampling points from the last decoder layer
q, ref = cap['q'], cap['ref']
print('ref shape', tuple(ref.shape))
with torch.no_grad():
    nh, nl, npnt = ca.n_heads, ca.n_levels, ca.n_points
    off = ca.sampling_offsets(q).view(1, q.shape[1], nh, nl, npnt, 2)
    aw = ca.attention_weights(q).view(1, q.shape[1], nh, nl * npnt).softmax(-1).view(1, q.shape[1], nh, nl, npnt)
    r = ref[:, :, None, :, None, :] if ref.shape[2] == nl else ref[:, :, None, :1, None, :]
    loc = r[..., :2] + off / npnt * r[..., 2:] * 0.5
print('heads', nh, 'levels', nl, 'points', npnt)
fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.9))
for ax, k in zip(axs, final[:2]):
    base(ax, f'負責「{lab(cl[k])}」的 query：只看 {nh*nl*npnt} 個點')
    L = loc[0, k].reshape(-1, 2).numpy(); A = aw[0, k].reshape(-1).numpy()
    rb = ref[0, k, 0].numpy(); cx, cy = rb[0] * W, rb[1] * H
    box(ax, xyxy(torch.tensor(rb)).numpy(), 'white', lw=2, ls='--')
    for (u, v), a in zip(L, A):
        ax.plot([cx, u * W], [cy, v * H], color='#ffd666', lw=.8, alpha=.6)
    ax.scatter(L[:, 0] * W, L[:, 1] * H, s=40 + 540 * A / A.max(), c='#ff4d6d', ec='white', lw=1, zorder=3)
    ax.scatter([cx], [cy], marker='*', s=260, c='#36cfc9', ec='black', zorder=4)
save(fig, '05_deform.png')
print('global tokens', (R // 14) ** 2)

# 6 Hungarian matching (stand-in labels = confident detections; candidates = top-8 queries shuffled)
gt = list(final); gtb = bx[gt]
cand = list(order[:8]); rng.shuffle(cand)

def giou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy; ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    cw = max(a[2], b[2]) - min(a[0], b[0]); ch = max(a[3], b[3]) - min(a[1], b[1])
    return inter / ua - (cw * ch - ua) / (cw * ch)

N = np.array([W, H, W, H])
C = np.zeros((len(cand), len(gt)))
for a, i in enumerate(cand):
    for b, j in enumerate(gt):
        C[a, b] = np.abs(bx[i] / N - gtb[b] / N).sum() * 5 - 2 * giou(bx[i], gtb[b]) - 2 * prob[i, cl[j]].item()
ri, ci = linear_sum_assignment(C)
fig, axs = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax = axs[0]; ax.imshow(C, cmap='viridis_r', aspect='auto')
ax.set_xticks(range(len(gt)), [f'標註{b+1}\n{lab(cl[j])}' for b, j in enumerate(gt)], fontsize=11)
ax.set_yticks(range(len(cand)), [f'query {i}' for i in cand], fontsize=11)
for a in range(len(cand)):
    for b in range(len(gt)): ax.text(b, a, f'{C[a,b]:.1f}', ha='center', va='center', color='white', fontsize=10)
for a, b in zip(ri, ci): ax.add_patch(P.Rectangle((b - .5, a - .5), 1, 1, fill=False, ec='#ff4d6d', lw=3.5))
ax.set_title('成本矩陣：數字越小越像（紅框 = 匈牙利匹配選中）', fontsize=13)
base(axs[1], '配對結果：灰色虛線 = 沒配到，要學說「沒東西」')
matched = set()
for a, b in zip(ri, ci):
    i = cand[a]; matched.add(i); box(axs[1], bx[i], cols[b % 6], text=f'q{i}→標註{b+1}')
for i in cand:
    if i not in matched: box(axs[1], bx[i], '#8c8c8c', lw=1.5, ls='--')
save(fig, '06_hungarian.png')
print('matched', [(int(cand[a]), int(b)) for a, b in zip(ri, ci)], 'unmatched', [int(i) for i in cand if i not in matched])

# 7 resolution: same weights, different input sizes
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, res in zip(axs, [280, 448, 560]):
    xx = TF.normalize(TF.to_tensor(img.resize((res, res))), m.means, m.stds)[None]
    with torch.no_grad(): oo = net(xx)
    s2, c2 = oo['pred_logits'][0].sigmoid().max(-1); b2 = xyxy(oo['pred_boxes'][0]).numpy()
    keep = [i for i in torch.argsort(-s2).tolist() if s2[i] > .5]
    base(ax, f'輸入 {res}×{res}：{(res//14)**2} 個 token，抓到 {len(keep)} 個')
    for k, i in enumerate(keep): box(ax, b2[i], cols[k % 6], text=f'{lab(c2[i].item())} {s2[i]:.2f}')
    print('res', res, [(name(c2[i].item()), round(float(s2[i]), 2)) for i in keep])
save(fig, '07_resolution.png')
