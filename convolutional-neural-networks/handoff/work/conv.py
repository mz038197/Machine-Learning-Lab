import torch, collections
from rfdetr import RFDETRBase
net = RFDETRBase().model.model
cnt = collections.Counter(); par = collections.Counter()
for n, m in net.named_modules():
    if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
        top = '.'.join(n.split('.')[:3]); cnt[top] += 1; par[top] += sum(p.numel() for p in m.parameters(recurse=False))
        if cnt[top] <= 2: print(n, m)
print(dict(cnt)); print(dict(par))
tot = sum(p.numel() for p in net.parameters()); print('total', tot, 'conv', sum(par.values()))
def sub(name): return sum(p.numel() for n,p in net.named_parameters() if n.startswith(name))
print('backbone.0', sub('backbone.0'), 'transformer', sub('transformer'))
print([type(m).__name__ for n,m in net.backbone[0].named_children()])
for n,m in net.backbone[0].named_modules():
    if 'projector' in n and n.count('.')<=2: print(n, type(m).__name__)
