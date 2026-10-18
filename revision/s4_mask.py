import numpy as np, pandas as pd
from common import *; from bodymask import body_mask
t=pd.read_csv("out_test_nn.csv"); X=load_images(); model=load_model()
G=np.stack([np.asarray(X[r]) for r in t.row.values]); M=np.stack([body_mask(g) for g in G])
np.save("out_test_bodymask.npy",M)
t["body_frac"]=M.reshape(len(M),-1).mean(1)
t["p_orig"]=predict(model,G)
t["p_body"]=predict(model,G*M)          # region outside the body set to 0 (air)
t["p_bg"]=predict(model,G*(1-M))        # body removed, only background/table/annotations kept
t.to_csv("out_test_mask.csv",index=False)
print(t.body_frac.describe())
