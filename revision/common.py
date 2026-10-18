"""Shared helpers for the round-2 revision analyses.

Re-implements, without changes, the model, Grad-CAM++ ROI extractor and handcrafted
descriptors used in XAI_Radiomics_ColonCancer_Study.ipynb, plus a corrected shape
calculation (area and perimeter from the same connected component).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, timm, cv2
from scipy import stats
from scipy.ndimage import label as scipy_label
from skimage.feature import graycomatrix, graycoprops

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
import os
DATA = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
torch.set_num_threads(2)


def load_meta():
    """Image table in the order used by the original loaders (classes sorted, Windows/NTFS file order)."""
    m = pd.read_csv(f"{DATA}/meta.csv")
    m["row"] = np.arange(len(m))  # row index into all_224.npy
    m["key"] = m["file"].str.lower()
    m = m.sort_values(["split", "label", "key"]).reset_index(drop=True)
    return m


def load_images():
    return np.load(f"{DATA}/all_224.npy", mmap_mode="r")


def to_tensor(gray224):
    """uint8 (N,224,224) grey -> normalized float tensor (N,3,224,224); identical to Resize+Normalize+ToTensorV2."""
    x = np.repeat(np.asarray(gray224, dtype=np.float32)[..., None] / 255.0, 3, axis=-1)
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


class ColonCancerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = "efficientnet_b0"
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False)
        nf = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(nf, 256), nn.ReLU(),
                                        nn.Dropout(0.2), nn.Linear(256, 2))

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def get_features(self, x):
        return self.backbone(x)


def load_model():
    m = ColonCancerClassifier()
    ck = torch.load(f"{DATA}/best_model_efficientnet_b0.pth", map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model_state_dict"])
    return m.eval()


@torch.no_grad()
def predict(model, gray, bs=64, with_features=False):
    probs, feats = [], []
    for i in range(0, len(gray), bs):
        x = to_tensor(gray[i:i + bs])
        f = model.get_features(x)
        p = torch.softmax(model.classifier(f), 1)[:, 1]
        probs.append(p.numpy())
        if with_features:
            feats.append(f.numpy())
    probs = np.concatenate(probs)
    return (probs, np.concatenate(feats)) if with_features else probs


# ---------------- ROI extraction (unchanged from the notebook) ----------------
def normalize_heatmap(h):
    h = np.array(h); h = h - h.min()
    return h / h.max() if h.max() > 0 else h


def extract_roi(heatmap, pct=85, min_area=100):
    thr = np.percentile(heatmap, pct)
    lab, n = scipy_label((heatmap >= thr).astype(np.uint8))
    full = np.ones_like(heatmap, dtype=np.uint8)
    if n == 0:
        return full, False
    areas = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(areas)) + 1  # first largest component, as in the original loop
    if areas[k - 1] < min_area:
        return full, False
    cc = (lab == k).astype(np.uint8)
    kern = np.ones((5, 5), np.uint8)
    roi = cv2.morphologyEx(cv2.morphologyEx(cc, cv2.MORPH_CLOSE, kern), cv2.MORPH_OPEN, kern)
    if roi.sum() == 0:
        return full, False
    return roi, True


# ---------------- handcrafted descriptors ----------------
def first_order(gray, mask):
    px = gray[mask > 0] if mask.sum() > 0 else gray.ravel()
    f = {}
    f["mean"] = np.mean(px); f["std"] = np.std(px); f["min"] = np.min(px); f["max"] = np.max(px)
    f["median"] = np.median(px)
    f["skewness"] = stats.skew(px) if len(px) > 2 else 0
    f["kurtosis"] = stats.kurtosis(px) if len(px) > 2 else 0
    f["entropy"] = stats.entropy(np.histogram(px, bins=64)[0] + 1e-10)
    f["energy"] = np.sum(px.astype(np.float64) ** 2) / len(px)
    f["range"] = f["max"] - f["min"]
    f["percentile_10"] = np.percentile(px, 10); f["percentile_25"] = np.percentile(px, 25)
    f["percentile_75"] = np.percentile(px, 75); f["percentile_90"] = np.percentile(px, 90)
    f["iqr"] = f["percentile_75"] - f["percentile_25"]
    return f


def shape_original(gray, mask):
    """Original implementation: area of all foreground pixels, perimeter of the FIRST external contour."""
    f = {}
    if mask.sum() > 10:
        f["area"] = mask.sum()
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cs):
            f["perimeter"] = cv2.arcLength(cs[0], True)
            f["compactness"] = 4 * np.pi * f["area"] / (f["perimeter"] ** 2 + 1e-10)
        else:
            f["perimeter"] = 0; f["compactness"] = 0
    else:
        f["area"] = gray.size; f["perimeter"] = 2 * sum(gray.shape); f["compactness"] = 1.0
    return f


def shape_corrected(gray, mask):
    """Corrected: area and perimeter from the SAME (largest) connected component of the final mask.
    Returns also the number of components so fragmented masks can be counted."""
    m = (mask > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    ncomp = n - 1
    if ncomp == 0:
        return {"area": 0.0, "perimeter": 0.0, "compactness": 0.0}, 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    comp = (lab == k).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = cv2.arcLength(max(cs, key=cv2.contourArea), True)
    area = float(comp.sum())
    return {"area": area, "perimeter": per, "compactness": 4 * np.pi * area / (per ** 2 + 1e-10)}, ncomp


def glcm(gray):
    g = graycomatrix(gray, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=256, symmetric=True, normed=True)
    return {f"glcm_{p}": graycoprops(g, p).mean() for p in
            ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]}


def denorm_gray(gray224):
    """Reproduces tensor_to_numpy + cv2.cvtColor((img*255).astype(uint8), RGB2GRAY) used in the notebook."""
    x = np.repeat(np.asarray(gray224, np.float32)[..., None] / 255.0, 3, -1)
    x = (x - MEAN) / STD
    x = np.clip(x * STD + MEAN, 0, 1)
    return cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
