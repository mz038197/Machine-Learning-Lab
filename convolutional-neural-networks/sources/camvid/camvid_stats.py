"""Count object-like blobs per class in CamVid (SegNet 11-class) annotations, per split and sequence."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1])
CLASSES = {2: "pole", 6: "sign/light", 8: "car", 9: "pedestrian", 10: "bicyclist"}
MIN_AREA = 30  # pixels; smaller blobs treated as label noise

for split in ("trainannot", "valannot"):
    files = sorted((root / split).glob("*.png"))
    seqs = Counter(f.name.split("_")[0] for f in files)
    print(f"\n=== {split}: {len(files)} frames, sequences {dict(seqs)}")
    blobs = defaultdict(list)  # class -> list of (frame, h, w, area, cy, cx)
    frames_with = Counter()
    for f in files:
        a = np.array(Image.open(f))
        for cid, name in CLASSES.items():
            lab, n = ndimage.label(a == cid)
            found = 0
            for sl, idx in zip(ndimage.find_objects(lab), range(1, n + 1)):
                area = int((lab[sl] == idx).sum())
                if area < MIN_AREA:
                    continue
                h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
                blobs[name].append((f.name, h, w, area, (sl[0].start + sl[0].stop) / 2, (sl[1].start + sl[1].stop) / 2))
                found += 1
            if found:
                frames_with[name] += 1
    for name in CLASSES.values():
        b = blobs[name]
        if not b:
            print(f"{name:11s} 0")
            continue
        hs = np.array([x[1] for x in b])
        print(
            f"{name:11s} blobs {len(b):5d} | frames with it {frames_with[name]:3d}/{len(files)}"
            f" | bbox height px median {int(np.median(hs)):3d}, <16px {np.mean(hs < 16):.0%}, <32px {np.mean(hs < 32):.0%}"
        )
    if split == "valannot":
        print("\nval frames with pedestrian blobs (area>=30):")
        per = Counter(x[0] for x in blobs["pedestrian"])
        print(sorted(per.items())[:15], "...", len(per), "frames")
        big = [x for x in blobs["pedestrian"] if x[1] >= 40]
        print("val pedestrians with height>=40px:", len(big), "in frames", sorted({x[0] for x in big})[:20])
        signs = sorted(blobs["sign/light"], key=lambda x: -x[3])
        np.save(root / "val_signs.npy", np.array([(x[0], x[1], x[2], x[3], x[4], x[5]) for x in signs], dtype=object), allow_pickle=True)
        print("val sign/light blobs saved:", len(signs))
