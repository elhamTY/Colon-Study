import numpy as np, cv2
def body_mask(g, thr=20, open_k=7, dil=2, min_area=300):
    """Body region of a 224x224 grey CT image.
    Pixels brighter than air (blurred value > 20) -> 7x7 elliptical opening, which detaches the
    scanner table, cushions and burned-in annotations -> connected components of at least 300 pixels and at least 5% of the largest
    -> convex hull of those components (so that internal or peripheral gas and lung are retained) -> 2-pixel dilation."""
    g=np.asarray(g)
    blur=cv2.GaussianBlur(g,(5,5),0)
    b=(blur>thr).astype(np.uint8)
    b=cv2.morphologyEx(b,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_k,open_k)))
    n,lab,st,_=cv2.connectedComponentsWithStats(b,8)
    if n<=1: return np.ones_like(g,np.uint8)
    areas=st[1:,cv2.CC_STAT_AREA]; keep=1+np.nonzero(areas>=max(min_area,0.05*areas.max()))[0]
    comp=np.isin(lab,keep).astype(np.uint8)
    cs,_=cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hull=cv2.convexHull(np.vstack(cs))
    m=np.zeros_like(g,np.uint8); cv2.fillPoly(m,[hull],1)
    if dil: m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dil+1,2*dil+1)))
    return m
