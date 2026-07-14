"""옵티컬 플로우로 실제 흐름 방향 진단: 중심 대비 방사 성분 평균.
양(+)=발산(전진), 음(-)=수렴(후진). VO 부호와 대조."""
import sys, cv2, numpy as np
V=sys.argv[1]; TAG=sys.argv[2]
cap=cv2.VideoCapture(V); N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
OSD=[(60,130,80,500),(950,1025,1600,1900)]
def getf(fi):
    cap.set(cv2.CAP_PROP_POS_FRAMES,fi); r,f=cap.read()
    if not r: return None
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    for y0,y1,x0,x1 in OSD: g[y0:y1,x0:x1]=int(g.mean())
    return cv2.resize(g,None,fx=0.5,fy=0.5)
pts=[int(N*x) for x in [0.1,0.2,0.3,0.4,0.55,0.7,0.85]]
print(f"{TAG} N={N}")
for fi in pts:
    g0=getf(fi); g1=getf(fi+8)
    if g0 is None or g1 is None: continue
    # 암부 중심
    bl=cv2.GaussianBlur(g0,(0,0),15); ys,xs=np.nonzero(bl<=np.percentile(bl,5))
    cx,cy=(xs.mean(),ys.mean()) if len(xs)>50 else (g0.shape[1]/2,g0.shape[0]/2)
    fl=cv2.calcOpticalFlowFarneback(g0,g1,None,0.5,3,25,3,5,1.2,0)
    H,W=g0.shape; yy,xx=np.mgrid[0:H,0:W]
    rx,ry=xx-cx,yy-cy; rn=np.sqrt(rx**2+ry**2)+1e-6
    radial=(fl[...,0]*rx+fl[...,1]*ry)/rn   # 방사 성분(양=발산)
    # 텍스처 있는 영역만(그래디언트)
    gm=np.abs(cv2.Sobel(g0.astype(np.float32),cv2.CV_32F,1,0,3))+np.abs(cv2.Sobel(g0.astype(np.float32),cv2.CV_32F,0,1,3))
    msk=(gm>np.percentile(gm,70))&(rn>30)&(rn<min(H,W)*0.45)
    rmean=float(np.median(radial[msk])) if msk.sum()>100 else float('nan')
    print(f"  f{fi}: center=({cx:.0f},{cy:.0f}) 방사플로우중앙={rmean:+.3f} ({'발산=전진' if rmean>0.02 else '수렴=후진' if rmean<-0.02 else '정지'})")
cap.release()
