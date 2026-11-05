import sys, torch, rfdetr, inspect
from PIL import Image
print(torch.__version__, rfdetr.__file__)
from rfdetr import RFDETRBase
m = RFDETRBase()
img = Image.open(sys.argv[1])
d = m.predict(img, threshold=0.5)
print(d.xyxy, d.class_id, d.confidence)
print([a for a in dir(m) if not a.startswith('_')])
print(type(m.model), [a for a in dir(m.model) if not a.startswith('_')])
print(getattr(m.model,'resolution',None))
