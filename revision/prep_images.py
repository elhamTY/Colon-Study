"""Step 0. Read every PNG in Split_Data/{train,val,test}/{Non_Cancer,cancer}, record metadata and hashes,
and store the 224x224 network-input images (grey; all files have R=G=B) for the revision analyses.

Usage: python prep_images.py --root "path/to/Split_Data" --out data
Resizing reproduces Albumentations Resize(224, 224) (cv2.INTER_LINEAR) applied to PIL .convert('RGB')."""
import os, csv, hashlib, argparse, numpy as np, cv2
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", default="data")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
rows, arrs = [], []
for split in ["test", "train", "val"]:
    for ci, c in enumerate(sorted(os.listdir(os.path.join(a.root, split)))):  # Non_Cancer=0, cancer=1
        for f in sorted(os.listdir(os.path.join(a.root, split, c)), key=str.lower):
            if not f.lower().endswith(".png"): continue
            p = os.path.join(a.root, split, c, f); raw = open(p, "rb").read()
            im = Image.open(p); W, H = im.size; rgb = np.array(im.convert("RGB"))
            eq = int((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all())
            arrs.append(cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)[..., 0])
            rows.append([split, c, f, ci, W, H, im.mode, eq, hashlib.md5(rgb.tobytes()).hexdigest(), hashlib.md5(raw).hexdigest()])
np.save(os.path.join(a.out, "all_224.npy"), np.stack(arrs))
with open(os.path.join(a.out, "meta.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["split", "cls", "file", "label", "W", "H", "mode", "rgb_equal", "pix_md5", "file_md5"]); w.writerows(rows)
print(len(rows), "images")
