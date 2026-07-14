"""전진 구간에서 flow divergence 부호 실측. baseline=24원본프레임."""
import sys, cv2, numpy as np
V=sys.argv[1]; TAG=sys.argv[2]
cap=cv2.VideoCapture(V); N=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
OSD=[(60,130,80,500),(950,1025,1600,1900)]
def getf(fi):
    cap.set(cv2.CAP_PROP_POS_FRAMES,fi); r,f=cap.read()
    if not r: return None
    g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    for y0,y1,x0,x1 in OSD: g[y0:y1,x0:x1]=int(g.mean())
    g=cv2.resize(g,None,fx=0.25,fy=0.25)  # DS=0.5 * flow 0.5
    return g
print(f"{TAG} N={N} (양=발산=전진 기대)")
for fr in [0.15,0.25,0.35,0.5,0.65,0.8]:
    fi=int(N*fr)
    g0=getf(fi); g1=getf(fi+24)
    if g0 is None or g1 is None: continue
    fl=cv2.calcOpticalFlowFarneback(g0,g1,None,0.5,3,21,3,5,1.2,0)
    du=cv2.Sobel(fl[:,:,0],cv2.CV_32F,1,0,3); dv=cv2.Sobel(fl[:,:,1],cv2.CV_32F,0,1,3)
    divg=du+dv
    gm=np.abs(cv2.Sobel(g0.astype(np.float32),cv2.CV_32F,1,0,3))+np.abs(cv2.Sobel(g0.astype(np.float32),cv2.CV_32F,0,1,3))
    tm=gm>np.percentile(gm,60)
    md=float(np.median(divg[tm])); mn=float(np.mean(divg[tm]))
    mag=float(np.median(np.abs(fl).sum(2)[tm]))
    print(f"  f{fi}: div중앙={md:+.4f} div평균={mn:+.4f} |flow|중앙={mag:.2f}px")
cap.release()
