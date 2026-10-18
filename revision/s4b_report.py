import numpy as np, pandas as pd
from stats_util import *
t=pd.read_csv("out_test_mask.csv"); y=t.label.values
for c in ["p_orig","p_body","p_bg"]:
    print(c, fmt(metrics(y,t[c])))
print("McNemar orig vs body:",mcnemar(y,t.p_orig,t.p_body)," orig vs bg:",mcnemar(y,t.p_orig,t.p_bg))
print("mean |dp| orig->body",np.abs(t.p_orig-t.p_body).mean(), " flips",((t.p_orig>=.5)!=(t.p_body>=.5)).sum())
# subsets
excl_exact = t.copy_in_other_split.fillna("")!=""
nd = t.nn_ssim<0.95
for name,sel in [("no exact copy in train/val",~excl_exact),("NN SSIM<0.95",nd),("NN SSIM<0.90",t.nn_ssim<0.90),("NN SSIM<0.80",t.nn_ssim<0.80)]:
    q=t[sel]; print(name, "| orig:",fmt(metrics(q.label,q.p_orig)))
    print("   body:",fmt(metrics(q.label,q.p_body))); print("   bg  :",fmt(metrics(q.label,q.p_bg)))
    print("   1-NN raw pixel acc:",(q.nn_label==q.label).mean(), "n",len(q))
print(t[["nn_ssim","label"]].groupby("label").describe())
