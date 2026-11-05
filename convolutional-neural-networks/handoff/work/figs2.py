import sys, os, numpy as np, torch, torchvision.transforms.functional as TF
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as P
from scipy.optimize import linear_sum_assignment
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
from rfdetr import RFDETRBase
from rfdetr.assets.coco_classes import COCO_CLASSES

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
img = Image.open(src).convert('RGB'); W, H = img.size
m = RFDETRBase(); net = m.model.model.eval()
R = 560; PS = 14; G = R // PS
im560 = img.resize((R, R))
x = TF.normalize(TF.to_tensor(im560), m.means, m.stds)[None]
cap = {}
ca = net.transformer.decoder.layers[-1].cross_attn
def pre(mod, args, kwargs):
    a = list(args) + [None] * 6
    cap['q'] = kwargs.get('query', a[0]); cap['ref'] = kwargs.get('reference_points', a[1])
ca.register_forward_pre_hook(pre, with_kwargs=True)
net.backbone[0].encoder.register_forward_hook(lambda mod, i, o: cap.__setitem__('feats', [t.detach() for t in o]))
with torch.no_grad(): o = net(x)

prob = o['pred_logits'][0].sigmoid(); score, cls = prob.max(-1)
sc = score.numpy(); cl = cls.numpy(); order = np.argsort(-sc)
zh = {'cat': '貓', 'remote': '遙控器', 'couch': '沙發'}
def lab(c): n = COCO_CLASSES.get(int(c), str(c)); return zh.get(n, n)
def xyxy(b, w=W, h=H):
    b = torch.as_tensor(b); cx, cy, bw, bh = b.unbind(-1)
    return torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1).numpy()
bx = xyxy(o['pred_boxes'][0])
final = [int(i) for i in order if sc[i] > 0.5]
cat1, cat2 = final[0], final[1]
cols = ['#ff7a45', '#36cfc9', '#ffd666', '#9254de', '#73d13d']
RED, YEL, CY = '#ff4d6d', '#ffd666', '#36cfc9'

def new(title, n=1, ratios=None):
    fig, axs = plt.subplots(1, n, figsize=(12.8, 7.2), gridspec_kw=dict(width_ratios=ratios) if ratios else None)
    fig.suptitle(title, fontsize=24, y=.97)
    return fig, (axs if n > 1 else [axs])
def base(ax, im=None, sub=None):
    im = img if im is None else im
    ax.imshow(im); ax.set_xlim(0, im.size[0]); ax.set_ylim(im.size[1], 0); ax.axis('off')
    if sub: ax.set_title(sub, fontsize=17)
def box(ax, b, c, lw=4, ls='-', alpha=1, text=None, fs=16):
    ax.add_patch(P.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=lw, ls=ls, alpha=alpha))
    if text: ax.text(b[0] + 4, b[1] + 4, text, color='black', fontsize=fs, va='top', bbox=dict(fc=c, ec='none', alpha=.92, pad=2))
def tag(ax, xy, t, c=RED, fs=18):
    ax.text(xy[0], xy[1], t, color='white', fontsize=fs, ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='circle', fc=c, ec='white', lw=2))
def save(fig, n):
    fig.subplots_adjust(top=.88, bottom=.04, left=.03, right=.97, wspace=.12)
    fig.savefig(os.path.join(out, n), dpi=100, facecolor='white'); plt.close(fig); print('saved', n)
rng = np.random.default_rng(0)

# ---------- Part 1: NMS ----------
fig, a = new('任務：找出照片裡的每個物件，框起來並標類別'); base(a[0]); save(fig, 'p01_task.png')

fig, a = new('傳統偵測器：每個位置都猜，同一隻貓被框 12 次（示意）'); base(a[0])
b = bx[cat1]; w, h = b[2] - b[0], b[3] - b[1]
for _ in range(12): box(a[0], b + rng.normal(0, 1, 4) * np.array([w, h, w, h]) * .07, cols[0], lw=2, alpha=.8)
save(fig, 'p02_dense.png')

fig, a = new('NMS：兩個框重疊太多，就刪掉分數低的那個（示意）'); base(a[0])
b1 = bx[cat1]; b2 = b1 + np.array([25, 15, 20, 10])
box(a[0], b1, cols[0], text='分數 0.96 → 留下'); box(a[0], b2, '#8c8c8c', ls='--', text='分數 0.80 → 刪掉')
a[0].text(W * .55, H * .9, '重疊比例 IoU 超過門檻（例如 0.5）就刪\n門檻要人手調', fontsize=18, color='white', bbox=dict(fc='black', alpha=.75))
save(fig, 'p03_nms_rule.png')

# ---------- Part 2: backbone ----------
fig, a = new('第 1 步：照片縮放成 560×560', 2)
base(a[0], sub=f'原圖 {W}×{H}'); base(a[1], im560, sub='縮放後 560×560'); save(fig, 'p04_resize.png')

fig, a = new(f'第 2 步：切成 {G}×{G} = {G*G} 個小方塊（patch）', 2, [1.4, 1])
base(a[0], im560)
for t in range(0, R + 1, PS): a[0].axhline(t, color='white', lw=.5, alpha=.8); a[0].axvline(t, color='white', lw=.5, alpha=.8)
pr, pc = 14, 10
a[0].add_patch(P.Rectangle((pc * PS, pr * PS), PS, PS, fill=False, ec=RED, lw=4))
crop = im560.crop((pc * PS, pr * PS, pc * PS + PS, pr * PS + PS))
a[1].imshow(crop, interpolation='nearest'); a[1].set_xticks([]); a[1].set_yticks([])
for s in a[1].spines.values(): s.set_color(RED); s.set_linewidth(5)
a[1].set_title('放大其中 1 塊：14×14 個像素\n→ 變成 1 個 token', fontsize=17)
save(fig, 'p05_patch.png')

fig, a = new('第 3 步：視窗注意力，切成 4 個視窗，各自在內部互看'); base(a[0], im560)
for t in range(0, R + 1, PS): a[0].axhline(t, color='white', lw=.4, alpha=.6); a[0].axvline(t, color='white', lw=.4, alpha=.6)
for k, (r0, c0) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
    a[0].add_patch(P.Rectangle((c0 * R / 2, r0 * R / 2), R / 2, R / 2, fc=cols[k], alpha=.3, ec=cols[k], lw=5))
    tag(a[0], (c0 * R / 2 + R / 4, r0 * R / 2 + R / 4), str(k + 1), cols[k], 26)
save(fig, 'p06_windows.png')

def pca_rgb(f):
    f = f[0].reshape(f.shape[1], -1).T.numpy(); f = f - f.mean(0)
    u, s, vt = np.linalg.svd(f, full_matrices=False); p = f @ vt[:3].T
    p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-6)
    return Image.fromarray((p.reshape(G, G, 3) * 255).astype(np.uint8)).resize((R, R), Image.NEAREST)
fig, a = new('DINOv2 骨幹看到什麼：每個 token 的特徵壓成顏色，顏色像 = 特徵像', 3)
base(a[0], im560, sub='原圖')
base(a[1], pca_rgb(cap['feats'][0]), sub='淺層（第 2 層）\n還在看顏色和紋理')
base(a[2], pca_rgb(cap['feats'][3]), sub='深層（第 11 層）\n貓、沙發、遙控器分開了')
save(fig, 'p07_dino.png')

# ---------- Part 3: queries ----------
enc = xyxy(o['enc_outputs']['pred_boxes'][0])
encs = o['enc_outputs']['pred_logits'][0].sigmoid().max(-1)[0].numpy()
ec = np.array([(b[0] + b[2]) / 2 for b in enc]), np.array([(b[1] + b[3]) / 2 for b in enc])
fig, a = new('第 4 步：先從特徵挑出 300 個初始猜測（每個點 = 一個猜測框的中心）'); base(a[0])
a[0].scatter(ec[0], ec[1], s=25 + 250 * encs / encs.max(), c=CY, ec='black', lw=.6, alpha=.85)
for i in np.argsort(-encs)[:3]: box(a[0], enc[i], CY, lw=2.5, ls='--')
a[0].text(8, H - 10, '點越大 = 初始分數越高   虛線 = 分數最高的 3 個初始框', fontsize=16, color='white', bbox=dict(fc='black', alpha=.75))
save(fig, 'p08_init.png')

stages = [enc, xyxy(o['aux_outputs'][0]['pred_boxes'][0]), xyxy(o['aux_outputs'][1]['pred_boxes'][0]), bx]
names = ['初始猜測', '解碼器第 1 層後', '第 2 層後', '第 3 層（最後）']
chg = [np.abs(stages[0][i] - bx[i]).sum() for i in final]; qk = final[int(np.argmax(chg))]
fb = bx[qk]; pad = 60
X0, Y0 = max(0, min(st[qk][0] for st in stages) - pad), max(0, min(st[qk][1] for st in stages) - pad)
X1, Y1 = min(W, max(st[qk][2] for st in stages) + pad), min(H, max(st[qk][3] for st in stages) + pad)
fig, axs = plt.subplots(2, 2, figsize=(12.8, 7.2)); axs = axs.ravel()
fig.suptitle(f'第 5 步：負責「{lab(cl[qk])}」的 query，每過一層解碼器就修正一次框', fontsize=22, y=.98)
for k, (st, nm) in enumerate(zip(stages, names)):
    ax = axs[k]; ax.imshow(img); ax.set_xlim(X0, X1); ax.set_ylim(Y1, Y0); ax.axis('off')
    d = np.abs(st[qk] - fb).mean()
    ax.set_title(f'{nm}：離最後的框平均差 {d:.0f} 像素', fontsize=15)
    if k: box(ax, stages[k - 1][qk], 'white', lw=2, ls='--')
    box(ax, st[qk], YEL, lw=3.5)
fig.subplots_adjust(top=.9, bottom=.02, left=.02, right=.98, hspace=.18, wspace=.05)
fig.savefig(os.path.join(out, 'p09_refine.png'), dpi=100, facecolor='white'); plt.close(fig)
print('refine query', qk, lab(cl[qk]), [round(float(np.abs(st[qk] - fb).mean()), 1) for st in stages])

fig, a = new('第 6 步：300 個 query 各交一個框（灰色 = 分數低）'); base(a[0])
for i in order[::-1]:
    if sc[i] < 0.5: box(a[0], bx[i], '#d9d9d9', lw=.7, alpha=.4)
for k, i in enumerate(final): box(a[0], bx[i], cols[k], text=f'{lab(cl[i])} {sc[i]:.2f}')
save(fig, 'p10_all300.png')

fig, a = new(f'300 個分數由高到低：只有 {len(final)} 個超過 0.5，{int((sc < .1).sum())} 個低於 0.1')
s = sc[order]; a[0].bar(range(300), s, color=[cols[min(k, 4)] if v > .5 else '#bfbfbf' for k, v in enumerate(s)], width=1)
a[0].axhline(.5, color=RED, ls='--', lw=2); a[0].text(120, .53, '門檻 0.5', color=RED, fontsize=18)
for k, i in enumerate(final): a[0].text(k * 9 + 12, .92 - k * .07, f'第{k+1}名 {lab(cl[i])} {sc[i]:.2f}', fontsize=16, color=cols[k], fontweight='bold')
a[0].set_xlabel('query 排名', fontsize=16); a[0].set_ylabel('信心分數', fontsize=16); a[0].set_ylim(0, 1); a[0].tick_params(labelsize=14)
save(fig, 'p11_scores.png')

fig, a = new('不用 NMS：直接留下分數 > 0.5 的框，就是答案'); base(a[0])
for k, i in enumerate(final): box(a[0], bx[i], cols[k], text=f'{lab(cl[i])} {sc[i]:.2f}')
save(fig, 'p12_result.png')

# ---------- Part 4: deformable attention ----------
q, ref = cap['q'], cap['ref']
with torch.no_grad():
    nh, nl, npn = ca.n_heads, ca.n_levels, ca.n_points
    off = ca.sampling_offsets(q).view(1, 300, nh, nl, npn, 2)
    aw = ca.attention_weights(q).view(1, 300, nh, nl * npn).softmax(-1).view(1, 300, nh, nl, npn)
    r = ref[:, :, None, :, None, :]
    loc = r[..., :2] + off / npn * r[..., 2:] * .5
def star(ax, k):
    rb = ref[0, k, 0].numpy(); return rb[0] * W, rb[1] * H, xyxy(rb)
fig, a = new(f'一般注意力：一個 query 要看全部 {G*G} 個 token（很慢）'); base(a[0])
cx, cy, _ = star(a[0], cat1)
gx, gy = np.meshgrid((np.arange(G) + .5) / G * W, (np.arange(G) + .5) / G * H)
for u, v in zip(gx.ravel(), gy.ravel()): a[0].plot([cx, u], [cy, v], color=YEL, lw=.3, alpha=.35)
a[0].scatter([cx], [cy], marker='*', s=600, c=CY, ec='black', zorder=4)
save(fig, 'p13_global.png')

for nm, k in [('p14_deform_a.png', cat1), ('p15_deform_b.png', cat2)]:
    fig, a = new(f'可變形注意力：負責這隻貓的 query 只看 {nh*nl*npn} 個點'); base(a[0])
    cx, cy, rb = star(a[0], k); box(a[0], rb, 'white', lw=2.5, ls='--')
    L = loc[0, k].reshape(-1, 2).numpy(); A = aw[0, k].reshape(-1).numpy()
    for (u, v) in L: a[0].plot([cx, u * W], [cy, v * H], color=YEL, lw=1.2, alpha=.7)
    a[0].scatter(L[:, 0] * W, L[:, 1] * H, s=60 + 700 * A / A.max(), c=RED, ec='white', lw=1.5, zorder=3)
    a[0].scatter([cx], [cy], marker='*', s=600, c=CY, ec='black', zorder=4)
    a[0].text(8, H - 10, '★ 猜測中心   白虛線 = 猜測範圍   紅點 = 取樣位置（越大越重要）', fontsize=16, color='white', bbox=dict(fc='black', alpha=.75))
    save(fig, nm)

# ---------- Part 5: Hungarian ----------
gt = final; gtb = bx[gt]
cand = [int(i) for i in order[:8]]; rng.shuffle(cand)
fig, a = new('訓練時：左邊是 8 個 query 的預測，右邊是 5 個人工標註（示意）', 2)
base(a[0], sub='8 個 query 的預測')
for k, i in enumerate(cand): box(a[0], bx[i], plt.cm.tab10(k), lw=2.5, text=f'q{i}', fs=13)
base(a[1], sub='標註（以本圖高信心結果代替）')
for b, j in enumerate(gt): box(a[1], gtb[b], cols[b], text=f'標註{b+1} {lab(cl[j])}', fs=13)
save(fig, 'p16_pred_gt.png')

def giou(p, g):
    ix = max(0, min(p[2], g[2]) - max(p[0], g[0])); iy = max(0, min(p[3], g[3]) - max(p[1], g[1]))
    inter = ix * iy; ua = (p[2] - p[0]) * (p[3] - p[1]) + (g[2] - g[0]) * (g[3] - g[1]) - inter
    cw = max(p[2], g[2]) - min(p[0], g[0]); ch = max(p[3], g[3]) - min(p[1], g[1])
    return inter / ua - (cw * ch - ua) / (cw * ch)
N = np.array([W, H, W, H]); C = np.zeros((8, len(gt)))
for r_, i in enumerate(cand):
    for c_, j in enumerate(gt):
        C[r_, c_] = np.abs(bx[i] / N - gtb[c_] / N).sum() * 5 - 2 * giou(bx[i], gtb[c_]) - 2 * prob[i, cl[j]].item()
ri, ci = linear_sum_assignment(C)
def matrix(ax, mark):
    ax.imshow(C, cmap='viridis_r', aspect='auto')
    ax.set_xticks(range(len(gt)), [f'標註{b+1}\n{lab(cl[j])}' for b, j in enumerate(gt)], fontsize=15)
    ax.set_yticks(range(8), [f'q{i}' for i in cand], fontsize=15)
    for r_ in range(8):
        for c_ in range(len(gt)): ax.text(c_, r_, f'{C[r_, c_]:.1f}', ha='center', va='center', color='white', fontsize=14)
    if mark:
        for r_, c_ in zip(ri, ci): ax.add_patch(P.Rectangle((c_ - .5, r_ - .5), 1, 1, fill=False, ec=RED, lw=5))
fig, a = new('成本表：每格 = 位置差 + 不重疊程度 − 類別分數，越小越像'); matrix(a[0], False); save(fig, 'p17_cost.png')
fig, a = new('匈牙利匹配：挑出總成本最低的一對一組合（紅框）'); matrix(a[0], True); save(fig, 'p18_match.png')
fig, a = new('配到的 5 個學修正框；沒配到的 3 個（灰虛線）學說「沒東西」'); base(a[0])
matched = set()
for r_, c_ in zip(ri, ci):
    i = cand[r_]; matched.add(i); box(a[0], bx[i], cols[c_], text=f'q{i} → 標註{c_+1}')
for i in cand:
    if i not in matched: box(a[0], bx[i], '#595959', lw=3, ls='--', text=f'q{i} → 沒東西', fs=13)
save(fig, 'p19_assign.png')
print('match', [(cand[r_], int(c_) + 1) for r_, c_ in zip(ri, ci)], 'unmatched', [i for i in cand if i not in matched])

# ---------- Part 6: resolution ----------
for res in [280, 448, 560]:
    xx = TF.normalize(TF.to_tensor(img.resize((res, res))), m.means, m.stds)[None]
    with torch.no_grad(): oo = net(xx)
    s2, c2 = oo['pred_logits'][0].sigmoid().max(-1); b2 = xyxy(oo['pred_boxes'][0])
    keep = [i for i in torch.argsort(-s2).tolist() if s2[i] > .5]
    fig, a = new(f'輸入 {res}×{res}：{(res//14)**2} 個 token，抓到 {len(keep)} 個物件'); base(a[0])
    for k, i in enumerate(keep): box(a[0], b2[i], cols[k], text=f'{lab(c2[i].item())} {s2[i]:.2f}')
    save(fig, f'p2{[280, 448, 560].index(res)}_res{res}.png')
    print('res', res, [(lab(c2[i].item()), round(float(s2[i]), 2)) for i in keep])
