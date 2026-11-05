from rfdetr import RFDETRBase
m = RFDETRBase()
net = m.model.model
print(m.model_config)
for n, mod in net.named_modules():
    if 'decoder' in n and n.count('.')<=4: print(n, type(mod).__name__)
