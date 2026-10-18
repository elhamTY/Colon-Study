"""Deterministic feature extraction: one row per original image, no augmentation, no resampling,
filenames retained. Grad-CAM++ ROI and descriptors as in the original study; shape descriptors are
computed both with the original implementation and with the corrected one (same component)."""
import numpy as np, pandas as pd, torch, time, sys, os
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from common import *
split=sys.argv[1]; START=int(sys.argv[2]); END=int(sys.argv[3])
m=load_meta(); X=load_images(); model=load_model()
s=m[m.split==split].reset_index(drop=True).iloc[START:END].reset_index(drop=True)
cam=GradCAMPlusPlus(model=model,target_layers=[model.backbone.conv_head])
rows=[]; deep=[]; t0=time.time(); BS=int(os.environ.get("BS","8"))
import gc, resource
for i in range(0,len(s),BS):
    g=np.stack([np.asarray(X[r]) for r in s.row.values[i:i+BS]])
    x=to_tensor(g)
    with torch.no_grad():
        f=model.get_features(x); p=torch.softmax(model.classifier(f),1)[:,1].numpy()
    deep.append(f.numpy())
    hm=cam(input_tensor=x,targets=[ClassifierOutputTarget(1)]*len(g))
    for j in range(len(g)):
        gray=denorm_gray(g[j]); heat=normalize_heatmap(hm[j])
        roi,ok=extract_roi(heat)
        whole=np.ones_like(gray,np.uint8)
        gl=glcm(gray); fw=first_order(gray,whole); fr=first_order(gray,roi)
        so_w=shape_original(gray,whole); so_r=shape_original(gray,roi)
        sc_w,_=shape_corrected(gray,whole); sc_r,ncomp=shape_corrected(gray,roi)
        r={"split":split,"file":s.cls[i+j]+"/"+s.file[i+j],"label":int(s.label[i+j]),"p_cancer":float(p[j]),
           "roi_ok":int(ok),"roi_ncomp":ncomp,"roi_area_frac":float(roi.mean())}
        for k,v in {**fw,**so_w,**gl}.items(): r["whole_"+k]=float(v)
        for k,v in {**fr,**so_r,**gl}.items(): r["xai_"+k]=float(v)
        for k,v in sc_w.items(): r["whole_"+k+"_corr"]=float(v)
        for k,v in sc_r.items(): r["xai_"+k+"_corr"]=float(v)
        rows.append(r)
    gc.collect()
    if (i//BS)%40==0: print(split,i+len(g),"/",len(s),round(time.time()-t0),"s","maxrss MB",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024,flush=True)
df=pd.DataFrame(rows); df.to_csv(f"parts/feat_{split}_{START:05d}.csv",index=False)
np.save(f"parts/deep_{split}_{START:05d}.npy",np.concatenate(deep)); print("DONE",split,df.shape,flush=True)
