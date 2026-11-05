import sys, os, json, numpy as np, torch, torch.nn.functional as Fn, torchvision.transforms.functional as TF
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as P
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False

src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
img = Image.open(src).convert('RGB'); W0, H0 = img.size
m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval()
S = 448; G = S // 14; N = G * G + 1
im = img.resize((S, S))
x_in = TF.normalize(TF.to_tensor(im), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])[None]
RED, CY, YEL = '#ff4d6d', '#36cfc9', '#ffd666'
cols = ['#ff7a45', '#36cfc9', '#ffd666', '#9254de', '#73d13d', '#ff4d6d']
info = {}

def new(title, n=1, ratios=None, rows=1):
    fig, axs = plt.subplots(rows, n, figsize=(12.8, 7.2), gridspec_kw=dict(width_ratios=ratios) if ratios else None)
    fig.suptitle(title, fontsize=23, y=.97); return fig, np.array(axs).ravel().tolist() if (n > 1 or rows > 1) else [axs]
def show(ax, a, sub=None, ec=None, interp='nearest'):
    ax.imshow(a, interpolation=interp); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(ec or 'white'); s.set_linewidth(5 if ec else 0)
    if sub: ax.set_title(sub, fontsize=15)
def save(fig, n, top=.87, **kw):
    fig.subplots_adjust(top=top, bottom=kw.get('bottom', .05), left=.03, right=.97, wspace=kw.get('ws', .1), hspace=kw.get('hs', .25))
    fig.savefig(os.path.join(out, n), dpi=100, facecolor='white'); plt.close(fig); print('saved', n)
def pca_img(t, ref=None):
    f = t.numpy(); f = f - f.mean(0)
    vt = ref if ref is not None else np.linalg.svd(f, full_matrices=False)[2][:3]
    p = f @ vt.T; p = (p - p.min(0)) / (p.max(0) - p.min(0) + 1e-6)
    return p.reshape(G, G, 3)

# ---------- forward pass, block by block ----------
with torch.no_grad():
    pe = m.patch_embed(x_in)                     # 1, 1024, 384
    x = m.prepare_tokens_with_masks(x_in)        # 1, 1025, 384
    xs, attns, ratios_a, ratios_m, hid = [x[0]], [], [], [], []
    for b in m.blocks:
        h = b.norm1(x)
        qkv = b.attn.qkv(h).reshape(1, N, 3, 6, 64).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        A = ((q * 64 ** -.5) @ k.transpose(-2, -1)).softmax(-1)
        o = b.attn.proj((A @ v).transpose(1, 2).reshape(1, N, 384))
        da = b.ls1(o); x1 = x + da
        hh = b.mlp.act(b.mlp.fc1(b.norm2(x1))); dm = b.ls2(b.mlp.fc2(hh)); x2 = x1 + dm
        assert torch.allclose(x2, b(x), atol=1e-3)
        ratios_a.append(float(da.norm(dim=-1).mean() / x.norm(dim=-1).mean()))
        ratios_m.append(float(dm.norm(dim=-1).mean() / x1.norm(dim=-1).mean()))
        attns.append(A[0]); hid.append(hh[0]); xs.append(x2[0]); x = x2
    final = m.norm(x)[0]
info['ratio_attn'] = [round(r, 3) for r in ratios_a]; info['ratio_mlp'] = [round(r, 3) for r in ratios_m]

ear = (262, 95)  # left cat head, original coords
er, ec_ = int(ear[1] / H0 * G), int(ear[0] / W0 * G); EI = 1 + er * G + ec_
ex, ey = (ec_ + .5) * 14, (er + .5) * 14

# r01
fig, a = new(f'輸入：照片縮放成 {S}×{S}，形狀 3 × {S} × {S}'); show(a[0], im, interp='bilinear'); save(fig, 'r01_input.png')

# r02 patch -> 588 -> 384
fig = plt.figure(figsize=(12.8, 7.2)); fig.suptitle('Patch Embedding：一塊 14×14 → 588 個數字 → 乘上權重 → 384 個數字', fontsize=21, y=.97)
a0 = fig.add_axes([.02, .1, .36, .72]); show(a0, im, interp='bilinear')
for t in range(0, S + 1, 14): a0.axhline(t, color='w', lw=.3, alpha=.6); a0.axvline(t, color='w', lw=.3, alpha=.6)
a0.add_patch(P.Rectangle((ec_ * 14, er * 14), 14, 14, fill=False, ec=RED, lw=3))
a1 = fig.add_axes([.42, .55, .14, .25]); show(a1, np.array(im)[er * 14:(er + 1) * 14, ec_ * 14:(ec_ + 1) * 14], '放大：14×14×3', RED)
raw = np.array(im)[er * 14:(er + 1) * 14, ec_ * 14:(ec_ + 1) * 14].reshape(-1) / 255
a2 = fig.add_axes([.6, .58, .37, .2]); a2.bar(range(588), raw, color='#8c8c8c', width=1); a2.set_title('攤平：588 個像素值', fontsize=15); a2.set_xlim(0, 588); a2.set_yticks([])
emb = pe[0, er * G + ec_].numpy()
a3 = fig.add_axes([.6, .15, .37, .25]); a3.bar(range(384), emb, color=[CY if v > 0 else RED for v in emb], width=1)
a3.set_title('Conv2d(3→384, kernel 14, stride 14) 之後：384 個數字', fontsize=15); a3.set_xlim(0, 384); a3.set_yticks([])
fig.text(.785, .46, '↓ 乘 588×384 權重矩陣', ha='center', fontsize=16)
fig.savefig(os.path.join(out, 'r02_patch_embed.png'), dpi=100, facecolor='white'); plt.close(fig); print('saved r02')

# r03 PCA of raw patch embeddings
fig, a = new(f'1024 個 patch 都做一次：得到 1024 × 384，壓成顏色看', 2)
show(a[0], im, '原圖', interp='bilinear'); show(a[1], pca_img(pe[0]), '剛 embedding 完：只反映顏色和亮度')
save(fig, 'r03_pe_pca.png')

# r04 sequence with CLS
fig = plt.figure(figsize=(12.8, 7.2)); fig.suptitle(f'加上 CLS token：1 + 1024 = {N} 個 token，每個 384 維', fontsize=22, y=.95)
ax = fig.add_axes([.02, .3, .96, .4]); ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis('off')
ax.add_patch(P.FancyBboxPatch((.2, .8), 1.2, 1.4, boxstyle='round,pad=.05', fc=YEL, ec='black', lw=2)); ax.text(.8, 1.5, 'CLS', ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(.8, .35, '代表整張圖\n（學出來的）', ha='center', fontsize=13)
for k in range(8):
    r, c = divmod(k, G); tile = np.array(im)[r * 14:(r + 1) * 14, c * 14:(c + 1) * 14]
    ax.imshow(tile, extent=(1.8 + k * 1.2, 2.8 + k * 1.2, .9, 2.1), interpolation='nearest')
    ax.text(2.3 + k * 1.2, .5, f'patch {k+1}', ha='center', fontsize=12)
ax.text(11.6, 1.5, '…', fontsize=40, ha='center', va='center'); ax.text(12.4, 1.5, '第\n1024', fontsize=14, ha='center', va='center')
fig.text(.5, .18, f'形狀：{N} × 384', ha='center', fontsize=22, color=RED)
fig.savefig(os.path.join(out, 'r04_cls.png'), dpi=100, facecolor='white'); plt.close(fig); print('saved r04')

# r05 position embeddings
pos = m.pos_embed[0, 1:].detach(); PG = int(round(pos.shape[0] ** .5)); posn = Fn.normalize(pos, dim=1)
fig, a = new(f'位置 embedding：{PG}×{PG} 個位置各有 384 維向量，看它和其他位置多像', 3)
for ax, (r, c), nm in zip(a, [(3, 3), (18, 18), (30, 8)], ['左上角', '正中央', '左下方']):
    s = (posn @ posn[r * PG + c]).reshape(PG, PG).numpy()
    ax.imshow(s, cmap='inferno'); ax.scatter([c], [r], marker='*', s=300, c=CY, ec='black'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{nm}的位置（★）：附近最亮', fontsize=15)
save(fig, 'r05_pos.png', top=.82)
info['pos_grid'] = PG

# r07 qkv for one token
with torch.no_grad():
    b0 = m.blocks[0]; qkv1 = b0.attn.qkv(b0.norm1(m.prepare_tokens_with_masks(x_in)))[0, EI].numpy()
fig, a = new('自注意力第 1 步：每個 token 的 384 維，乘權重變成 1152 維，切成 Q、K、V', 1)
a[0].axis('off')
ax = fig.add_axes([.06, .55, .88, .22]); ax.bar(range(1152), qkv1, width=1, color=['#ff7a45'] * 384 + ['#36cfc9'] * 384 + ['#9254de'] * 384)
ax.set_xlim(0, 1152); ax.set_yticks([]); ax.set_xticks([192, 576, 960], ['Q（我要找什麼）384', 'K（我是什麼）384', 'V（我帶什麼內容）384'], fontsize=15)
ax.set_title('左貓頭那個 token 在第 1 層算出的 1152 個數字', fontsize=15)
ax2 = fig.add_axes([.06, .12, .88, .25]); ax2.set_xlim(0, 6); ax2.set_ylim(0, 1); ax2.axis('off')
for h in range(6):
    ax2.add_patch(P.FancyBboxPatch((h + .08, .25), .84, .5, boxstyle='round,pad=.02', fc=cols[h], alpha=.5, ec='black'))
    ax2.text(h + .5, .5, f'頭 {h+1}\n64 維', ha='center', va='center', fontsize=15)
ax2.text(3, .9, 'Q、K、V 各自再切成 6 個頭，每個頭 64 維，各算各的注意力', ha='center', fontsize=16)
fig.savefig(os.path.join(out, 'r07_qkv.png'), dpi=100, facecolor='white'); plt.close(fig); print('saved r07')

# r08 attention of ear token, 6 heads, layer 1
def att_map(L, h, i=EI): return attns[L][h, i, 1:].reshape(G, G).numpy()
fig, a = new('第 1 層：左貓頭的 token（★）在 6 個頭分別看哪裡（越亮越注意）', 3, rows=2)
for h in range(6):
    a[h].imshow(im); a[h].imshow(np.kron(att_map(0, h), np.ones((14, 14))), cmap='inferno', alpha=.65)
    a[h].scatter([ex], [ey], marker='*', s=250, c=CY, ec='black'); a[h].axis('off'); a[h].set_title(f'頭 {h+1}', fontsize=15)
save(fig, 'r08_heads_l1.png', top=.88, hs=.15, ws=.02)

# r09 MLP hidden
hv = hid[5][EI].numpy()
fig, a = new('MLP：384 → 放大成 1536 → GELU → 壓回 384', 2, [1, 2.2])
xx = np.linspace(-4, 3, 200); a[0].plot(xx, Fn.gelu(torch.tensor(xx)).numpy(), color=RED, lw=3); a[0].axhline(0, color='gray', lw=.5); a[0].axvline(0, color='gray', lw=.5)
a[0].set_title('GELU：負的變接近 0', fontsize=16); a[0].tick_params(labelsize=12)
a[1].bar(range(1536), hv, width=1, color=['#ff7a45' if v > .5 else '#bfbfbf' for v in hv]); a[1].set_xlim(0, 1536)
a[1].set_title(f'第 6 層，左貓頭 token 的 1536 個中間值：只有 {int((hv > .5).sum())} 個大於 0.5', fontsize=15); a[1].tick_params(labelsize=12)
save(fig, 'r09_mlp.png', top=.85, ws=.15)
info['mlp_active'] = int((hv > .5).sum())

# r11 PCA per layer (12)
fig, a = new('12 個 Block 逐層看：每層輸出壓成顏色（顏色像 = 特徵像）', 6, rows=2)
for L in range(12):
    show(a[L], pca_img(xs[L + 1][1:]), f'第 {L+1} 層')
save(fig, 'r11_layers_pca.png', top=.86, hs=.0, ws=.04, bottom=.12)

# r12 similarity to ear token across layers
fig, a = new('左貓頭的 token（★）和其他 token 多像：第 1、4、8、12 層', 4)
for ax, L in zip(a, [1, 4, 8, 12]):
    f = Fn.normalize(xs[L][1:], dim=1); s = (f @ f[EI - 1]).reshape(G, G).numpy()
    ax.imshow(im); ax.imshow(np.kron(s, np.ones((14, 14))), cmap='inferno', alpha=.65); ax.scatter([ex], [ey], marker='*', s=250, c=CY, ec='black'); ax.axis('off')
    ax.set_title(f'第 {L} 層', fontsize=17)
save(fig, 'r12_sim_layers.png', top=.8, ws=.03)

# r13 CLS attention, last layer, 6 heads
fig, a = new('第 12 層：CLS token 在 6 個頭分別看哪裡', 3, rows=2)
for h in range(6):
    s = attns[11][h, 0, 1:].reshape(G, G).numpy()
    a[h].imshow(im); a[h].imshow(np.kron(s, np.ones((14, 14))), cmap='inferno', alpha=.7); a[h].axis('off'); a[h].set_title(f'頭 {h+1}', fontsize=15)
save(fig, 'r13_cls_attn.png', top=.88, hs=.15, ws=.02)

# r14 outputs
fig, a = new('最後一層 LayerNorm 之後，得到兩種輸出', 2, [1, 1.3])
a[0].bar(range(384), final[0].numpy(), width=1, color=YEL); a[0].set_xlim(0, 384); a[0].set_yticks([])
a[0].set_title('CLS 特徵：1 × 384\n代表整張圖，用於分類、找相似圖', fontsize=15)
show(a[1], pca_img(final[1:]), f'patch 特徵：{G*G} × 384（{G}×{G} 格）\n每格一個向量，用於偵測、分割')
save(fig, 'r14_outputs.png', top=.8, ws=.15)

# r16 RF-DETR windowed variant (from rfdetr source: out_feature_indexes=[2,5,8,11])
fig, a = new('RF-DETR Base 裡的 DINOv2：12 層怎麼分工', 1); ax = a[0]; ax.set_xlim(0, 12.5); ax.set_ylim(0, 4); ax.axis('off')
outs = [2, 5, 8, 11]; glob = [3, 6, 9, 12]
for L in range(1, 13):
    g = L in glob; xL = L - .5
    ax.add_patch(P.FancyBboxPatch((xL - .4, 1.6), .8, 1.2, boxstyle='round,pad=.03', fc=RED if g else CY, alpha=.85, ec='black'))
    ax.text(xL, 2.2, f'{L}', ha='center', va='center', fontsize=20, color='white', fontweight='bold')
    ax.text(xL, 3.0, '全域' if g else '視窗', ha='center', fontsize=13)
    if L in outs:
        ax.annotate('', xy=(xL + .5, .8), xytext=(xL + .5, 1.55), arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))
for k, L in enumerate(outs): ax.text(L, .45, f'第 {L} 層輸出', ha='center', fontsize=13)
ax.text(6.25, .0, '↓ 4 份 40×40×384 特徵合併後送進投影器（輸入 560×560 時）', ha='center', fontsize=16)
ax.text(.1, 3.7, '■ 青色 = 視窗注意力（切 4 塊，各自互看）   ■ 紅色 = 全域注意力（全部 token 互看）', fontsize=15)
save(fig, 'r16_rfdetr_layers.png', top=.88)

json.dump(info, open(os.path.join(out, 'info.json'), 'w'), indent=1); print(info)
