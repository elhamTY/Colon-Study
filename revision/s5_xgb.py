import numpy as np, pandas as pd, xgboost as xgb, json, sys
from sklearn.preprocessing import StandardScaler
from stats_util import *
HC=["mean","std","min","max","median","skewness","kurtosis","entropy","energy","range",
    "percentile_10","percentile_25","percentile_75","percentile_90","iqr","area","perimeter","compactness",
    "glcm_contrast","glcm_dissimilarity","glcm_homogeneity","glcm_energy","glcm_correlation"]
def fit(Xtr,ytr,Xte):
    Xtr=np.nan_to_num(Xtr); Xte=np.nan_to_num(Xte)
    sc=StandardScaler().fit(Xtr)
    clf=xgb.XGBClassifier(n_estimators=100,max_depth=5,learning_rate=0.1,random_state=42,eval_metric="logloss")
    clf.fit(sc.transform(Xtr),ytr); return clf.predict_proba(sc.transform(Xte))[:,1]
def strategies(D,W,R):
    return {"Deep only":D,"Whole-image handcrafted":W,"ROI-associated handcrafted":R,
            "Deep + whole-image handcrafted":np.hstack([D,W]),"Deep + ROI-associated handcrafted":np.hstack([D,R]),
            "All features":np.hstack([D,W,R])}
def run(tr,te,label):
    out={}; Str=strategies(*tr[:3]); Ste=strategies(*te[:3])
    for k in Str: out[k]=fit(Str[k],tr[3],Ste[k])
    return out
# ---- (A) check: original sampled/augmented exports -> should reproduce manuscript Table 3
if "--check" in sys.argv:
    a=pd.read_csv("data/train_all_features_combined.csv"); b=pd.read_csv("data/test_all_features_combined.csv")
    def split(df): return (df[[f"deep_{i}" for i in range(1280)]].values, df[["whole_"+c for c in HC]].values, df[["xai_"+c for c in HC]].values, df.label.values)
    r=run(split(a),split(b),"orig"); y=b.label.values
    import json
    json.dump({k:{kk:(float(vv) if not isinstance(vv,tuple) else [float(x) for x in vv]) for kk,vv in metrics(y,p).items()} for k,p in r.items()},open("results_xgb_orig.json","w"),indent=1)
    sys.exit()
# ---- (B) deterministic re-extraction
import glob
def load(split):
    fs=sorted(glob.glob(f"parts/feat_{split}_*.csv")); ds=sorted(glob.glob(f"parts/deep_{split}_*.npy"))
    return pd.concat([pd.read_csv(f) for f in fs],ignore_index=True), np.concatenate([np.load(d) for d in ds])
tr,Dtr=load("train"); te,Dte=load("test")
assert len(tr)==7937 and len(te)==1702 and tr.file.is_unique and te.file.is_unique
def hc(df,pref,corr):
    cols=[pref+c+("_corr" if corr and c in ("area","perimeter","compactness") else "") for c in HC]; return df[cols].values
res={}
for tag,corr in [("det_origshape",False),("det_corrshape",True)]:
    r=run((Dtr,hc(tr,"whole_",corr),hc(tr,"xai_",corr),tr.label.values),(Dte,hc(te,"whole_",corr),hc(te,"xai_",corr)),tag)
    res[tag]=r
    for k,p in r.items(): print(tag,k,fmt(metrics(te.label.values,p)))
# save predictions
P=te[["file","label","p_cancer"]].copy()
for tag in res:
    for k,p in res[tag].items(): P[f"{tag}|{k}"]=p
P.to_csv("out_xgb_test_preds.csv",index=False)
# ---- summary JSON for the manuscript
import json
y=te.label.values; J={}
cnn=te.p_cancer.values
J["cnn"]=metrics(y,cnn)
for tag in res:
    J[tag]={k:metrics(y,p) for k,p in res[tag].items()}
    J[tag+"_mcnemar_roi_vs_whole"]=mcnemar(y,res[tag]["ROI-associated handcrafted"],res[tag]["Whole-image handcrafted"])
    J[tag+"_mcnemar_deep_vs_cnn"]=mcnemar(y,res[tag]["Deep only"],cnn)
    for k in ["Deep + whole-image handcrafted","Deep + ROI-associated handcrafted","All features"]:
        J[tag+"_mcnemar_fusion_vs_deep|"+k]=mcnemar(y,res[tag][k],res[tag]["Deep only"])
# de-duplicated test subset (no identical copy in train/val)
nn=pd.read_csv("out_test_nn.csv"); nn["file2"]=nn.cls+"/"+nn.file
keep=~te.file.map(dict(zip(nn.file2,nn.copy_in_other_split.fillna("")!=""))).astype(bool).values
J["dedup_n"]=int(keep.sum())
J["dedup"]={k:metrics(y[keep],p[keep]) for k,p in res["det_corrshape"].items()}
J["dedup_cnn"]=metrics(y[keep],cnn[keep])
J["roi_fragmented_train"]=int((tr.roi_ncomp>1).sum()); J["roi_fragmented_test"]=int((te.roi_ncomp>1).sum())
J["roi_fallback_train"]=int((tr.roi_ok==0).sum()); J["roi_fallback_test"]=int((te.roi_ok==0).sum())
J["roi_area_frac_median"]=float(pd.concat([tr,te]).roi_area_frac.median())
J["train_counts"]=tr.label.value_counts().to_dict()
def conv(o):
    if isinstance(o,dict): return {str(k):conv(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [conv(v) for v in o]
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    return o
json.dump(conv(J),open("results_xgb.json","w"),indent=1)
print("saved results_xgb.json")
