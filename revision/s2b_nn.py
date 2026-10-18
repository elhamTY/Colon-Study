import numpy as np, pandas as pd, cv2
from skimage.metrics import structural_similarity as ssim
from common import *
m=pd.read_csv("out_meta_sorted.csv"); X=load_images()
# exact-duplicate status of each image
md5split=m.groupby("pix_md5")["split"].apply(set)
def other(r): return sorted(md5split[r.pix_md5]-{r.split}) if len(md5split[r.pix_md5])>1 or (m.pix_md5==r.pix_md5).sum()>1 else []
m["n_copies"]=m.groupby("pix_md5")["file"].transform("size")-1
m["copy_in_other_split"]=[ "+".join(sorted(md5split[h]-{s})) for h,s in zip(m.pix_md5,m.split)]
for s in ["train","val","test"]:
    q=m[m.split==s]; print(s,"images",len(q),"with >=1 identical copy anywhere",(q.n_copies>0).sum(),"with copy in another partition",(q.copy_in_other_split!="").sum())
t=m[m.split=="test"]
print("test images with identical copy in TRAIN:",t.copy_in_other_split.str.contains("train").sum(),
      " in train or val:",(t.copy_in_other_split!="").sum())
print(pd.crosstab(t.cls,t.copy_in_other_split!=""))
# label conflicts among identical copies
conf=m.groupby("pix_md5")["label"].nunique(); print("groups with conflicting labels:",(conf>1).sum())
# nearest neighbour of each test image among train+val, 64x64 grey, Pearson correlation
def small(rows): return np.stack([cv2.resize(np.asarray(X[r]),(64,64),interpolation=cv2.INTER_AREA) for r in rows]).reshape(len(rows),-1).astype(np.float32)
def z(a): a=a-a.mean(1,keepdims=True); return a/ (np.linalg.norm(a,axis=1,keepdims=True)+1e-6)
tr=m[m.split!="test"].reset_index(drop=True)
A=z(small(t.row.values)); B=z(small(tr.row.values))
C=A@B.T
nn=C.argmax(1); t=t.copy()
t["nn_corr"]=C.max(1); t["nn_file"]=(tr.split+"/"+tr.cls+"/"+tr.file).values[nn]; t["nn_label"]=tr.label.values[nn]
t["nn_ssim"]=[ssim(np.asarray(X[a]),np.asarray(X[b]),data_range=255) for a,b in zip(t.row.values,tr.row.values[nn])]
t["nn_mae"]=[np.abs(np.asarray(X[a]).astype(float)-np.asarray(X[b])).mean() for a,b in zip(t.row.values,tr.row.values[nn])]
print("1-NN raw-pixel classifier accuracy on test:",(t.nn_label==t.label).mean())
for th in [1.0,0.99,0.98,0.95,0.90,0.80]:
    print(f"nn_ssim>={th}: {(t.nn_ssim>=th-1e-9).sum()} ({(t.nn_ssim>=th-1e-9).mean()*100:.1f}%)")
print(t.nn_ssim.describe())
# within-class nearest neighbour for non-trivial images
t.to_csv("out_test_nn.csv",index=False); m.to_csv("out_meta_dups.csv",index=False)
