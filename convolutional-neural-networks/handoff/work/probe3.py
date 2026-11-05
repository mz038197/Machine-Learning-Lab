import torch
from rfdetr import RFDETRBase
m = RFDETRBase(); net = m.model.model.eval()
print([n for n,_ in net.named_children()])
print(type(net.backbone).__name__, [n for n,_ in net.backbone.named_children()])
def show(o, p=''):
    if torch.is_tensor(o): print(p, tuple(o.shape))
    elif isinstance(o,(list,tuple)): [show(x, p+f'[{i}]') for i,x in enumerate(o)]
    elif isinstance(o,dict): [show(v, p+f'.{k}') for k,v in o.items()]
    else: print(p, type(o).__name__, [a for a in dir(o) if not a.startswith('_')][:8])
net.backbone.register_forward_hook(lambda mod,i,o: show(o,'backbone_out'))
net.backbone[0].encoder.register_forward_hook(lambda mod,i,o: show(o,'encoder_out')) if hasattr(net.backbone[0],'encoder') else print('children0',[n for n,_ in net.backbone[0].named_children()])
for i,l in enumerate(net.transformer.decoder.layers):
    l.cross_attn.register_forward_pre_hook(lambda mod,a,k,i=i: print('layer',i,'ref',tuple(k.get('reference_points',a[1] if len(a)>1 else None).shape), k.get('reference_points',a[1] if len(a)>1 else None)[0,0,0]), with_kwargs=True)
with torch.no_grad(): o = net(torch.randn(1,3,560,560))
show(o,'out')
