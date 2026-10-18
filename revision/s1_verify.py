import numpy as np, pandas as pd, time
from common import *
m=load_meta(); X=load_images(); model=load_model()
t=m[m.split=="test"].reset_index(drop=True)
t0=time.time(); p,f=predict(model, X[t.row.values], with_features=True); print("sec",time.time()-t0)
pred=(p>=0.5).astype(int); y=t.label.values
print("errors",(pred!=y).sum(),"FP",((pred==1)&(y==0)).sum(),"FN",((pred==0)&(y==1)).sum())
from sklearn.metrics import roc_auc_score; print("AUC",roc_auc_score(y,p))
t["p_cancer"]=p; t.to_csv("out_test_preds_original.csv",index=False); np.save("out_test_deep.npy",f)
old=pd.read_csv(f"{DATA}/test_all_features_combined.csv")
od=old[[f"deep_{i}" for i in range(1280)]].values
print("label order match:",(old.label.values==y).all())
d=np.abs(od-f).max(1); print("deep feature max abs diff: median",np.median(d),"max",d.max())
print(t[pred!=y][["cls","file","p_cancer"]])
