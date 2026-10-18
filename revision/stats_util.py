import numpy as np
from scipy.stats import norm, binomtest
from sklearn.metrics import roc_auc_score
def wilson(k,n,a=0.05):
    if n==0: return (np.nan,np.nan)
    z=norm.ppf(1-a/2); p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)
def hanley(auc,n1,n0,a=0.05):
    q1=auc/(2-auc); q2=2*auc*auc/(1+auc)
    se=np.sqrt((auc*(1-auc)+(n1-1)*(q1-auc*auc)+(n0-1)*(q2-auc*auc))/(n1*n0)); z=norm.ppf(1-a/2)
    return (max(0,auc-z*se),min(1,auc+z*se))
def metrics(y,p,thr=0.5):
    y=np.asarray(y); p=np.asarray(p); yh=(p>=thr).astype(int)
    tp=int(((yh==1)&(y==1)).sum()); tn=int(((yh==0)&(y==0)).sum()); fp=int(((yh==1)&(y==0)).sum()); fn=int(((yh==0)&(y==1)).sum())
    n=len(y); acc=(tp+tn)/n
    r=dict(n=n,tp=tp,tn=tn,fp=fp,fn=fn,acc=acc,acc_ci=wilson(tp+tn,n),
           sens=tp/(tp+fn) if tp+fn else np.nan, sens_ci=wilson(tp,tp+fn),
           spec=tn/(tn+fp) if tn+fp else np.nan, spec_ci=wilson(tn,tn+fp),
           prec=tp/(tp+fp) if tp+fp else np.nan, f1=2*tp/(2*tp+fp+fn) if tp else 0.0)
    if len(set(y))==2:
        auc=roc_auc_score(y,p); r["auc"]=auc; r["auc_ci"]=hanley(auc,int(y.sum()),int((1-y).sum()))
    return r
def mcnemar(y,pa,pb,thr=0.5):
    a=(np.asarray(pa)>=thr)==y; b=(np.asarray(pb)>=thr)==y
    n01=int((a&~b).sum()); n10=int((~a&b).sum())
    p=binomtest(min(n01,n10),n01+n10,0.5).pvalue if n01+n10 else 1.0
    return n01,n10,p
def fmt(r):
    s=f"n={r['n']} TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']} acc={r['acc']*100:.2f} ({r['acc_ci'][0]*100:.2f}-{r['acc_ci'][1]*100:.2f}) sens={r['sens']*100:.2f} spec={r['spec']*100:.2f} f1={r['f1']*100:.2f}"
    if 'auc' in r: s+=f" auc={r['auc']:.4f} ({r['auc_ci'][0]:.4f}-{r['auc_ci'][1]:.4f})"
    return s
