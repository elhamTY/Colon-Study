import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import *
t=pd.read_csv("out_test_mask.csv"); X=load_images(); M=np.load("out_test_bodymask.npy")
# two cancer-labelled and two non-cancer-labelled examples, one of each changed by masking
pick=[]
for lab in [1,0]:
    q=t[t.label==lab]
    flip=q[((q.p_orig>=.5)!=(q.p_body>=.5))]; keep=q[((q.p_orig>=.5)==(q.p_body>=.5))]
    pick+= [flip.index[0], keep.index[len(keep)//2]]
fig,ax=plt.subplots(len(pick),3,figsize=(9,3*len(pick)))
for i,ix in enumerate(pick):
    g=np.asarray(X[t.row[ix]]); m=M[ix]
    for j,(img,ttl,pc) in enumerate([(g,"Original",t.p_orig[ix]),(g*m,"Outside body masked",t.p_body[ix]),(g*(1-m),"Body removed",t.p_bg[ix])]):
        ax[i,j].imshow(img,cmap="gray",vmin=0,vmax=255); ax[i,j].axis("off")
        ax[i,j].set_title(f"{ttl}\nP(cancer) = {pc:.3f}",fontsize=9)
    ax[i,0].text(-12,112,"Cancer-labelled" if t.label[ix]==1 else "Non-cancer-labelled",rotation=90,va="center",ha="right",fontsize=9)
plt.tight_layout(); plt.savefig("Figure7_masking.png",dpi=300); plt.savefig("Figure7_masking_preview.png",dpi=60)
