import numpy as np, pandas as pd, imagehash
from PIL import Image
from common import *
m=load_meta(); X=load_images()
# exact duplicates (identical decoded pixels of the ORIGINAL 512x512 files)
g=m.groupby("pix_md5")
m["dup_group_size"]=g["file"].transform("size")
ex=m[m.dup_group_size>1]
print("exact: images in duplicate groups",len(ex),"groups",ex.pix_md5.nunique())
grp=ex.groupby("pix_md5").agg(splits=("split",lambda s:"+".join(sorted(s))),classes=("cls",lambda s:"+".join(sorted(set(s)))),n=("file","size"))
print(grp.splits.value_counts()); print(grp.classes.value_counts())
print("file-level md5 groups:", (m.groupby("file_md5").size()>1).sum())
# perceptual hash on 224 grey
H=np.array([int(str(imagehash.phash(Image.fromarray(np.asarray(X[r])),hash_size=8)),16) for r in m.row.values],dtype=np.uint64)
np.save("out_phash.npy",H); m.to_csv("out_meta_sorted.csv",index=False)
# all-pairs Hamming distance
def popcnt(a):
    a=a.copy(); c=np.zeros(a.shape,np.uint8)
    for _ in range(64): c+= (a & np.uint64(1)).astype(np.uint8); a>>=np.uint64(1)
    return c
N=len(H); rows=[]
for i in range(0,N,400):
    d=popcnt(H[i:i+400,None]^H[None,:])
    ii,jj=np.nonzero(d<=10)
    keep=jj>ii+i
    for a,b in zip(ii[keep]+i,jj[keep]): rows.append((a,b,int(d[a-i,b])))
P=pd.DataFrame(rows,columns=["i","j","ham"])
P.to_csv("out_pairs_ham10.csv",index=False)
P["si"]=m.split.values[P.i]; P["sj"]=m.split.values[P.j]
P["cross"]=P.si!=P.sj
for t in [0,2,4,6,8,10]:
    q=P[P.ham<=t]; print("ham<=",t,"pairs",len(q),"cross-partition pairs",q.cross.sum())
