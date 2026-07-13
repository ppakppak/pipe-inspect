"""조인트(이음부) 검출 프로토타입 — VO 척추의 절대 눈금자 후보.

원리: 조인트는 관 전둘레에 걸친 링 구조 → 전방뷰에서 특정 반경의 전주(全周)
고에너지 원호로 나타남. 밴드(θ×v)에서 '거의 모든 θ에서 동시에 강한' 행(v)을
찾고, 그 이벤트가 시간축에서 반복되는지(관 길이 주기) 확인.

출력: 프레임별 최대 전주에너지·해당 v, 그리고 VO 체인리지와 결합할 이벤트 후보.
usage: joint_detect.py <video> <tag> [stop_frame]
"""
import os
import sys

import cv2
import numpy as np

V = sys.argv[1]
TAG = sys.argv[2]
STOP = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
OUT = "/tmp/vo_test"
os.makedirs(OUT, exist_ok=True)

STRIDE = 5
NTH, NV = 192, 100
RHO_IN = 45.0
DS = 0.5
OSD_RECTS = [(60, 130, 80, 500), (950, 1025, 1600, 1900)]

cap = cv2.VideoCapture(V)
N = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), STOP)
th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
cos_t, sin_t = np.cos(th), np.sin(th)
center = None
rows = []
fi = -1
while True:
    ret, fr = cap.read()
    if not ret or fi + 1 >= N:
        break
    fi += 1
    if fi % STRIDE:
        continue
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    for (y0, y1, x0, x1) in OSD_RECTS:
        g[y0:y1, x0:x1] = int(g.mean())
    g = cv2.resize(g, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)
    H, W = g.shape
    blur = cv2.GaussianBlur(g, (0, 0), 15)
    ys, xs = np.nonzero(blur <= np.percentile(blur, 5))
    if len(xs) >= 50:
        c_new = np.array([xs.mean(), ys.mean()])
        center = c_new if center is None else 0.9 * center + 0.1 * c_new
    if center is None:
        center = np.array([W / 2.0, H / 2.0])
    cx, cy = center
    rho_out = 0.9 * min(H, W) / 2.0
    v = np.linspace(1.0 / rho_out, 1.0 / RHO_IN, NV)
    rho = 1.0 / v
    mapx = (cx + rho[:, None] * cos_t[None, :]).astype(np.float32)
    mapy = (cy + rho[:, None] * sin_t[None, :]).astype(np.float32)
    valid = ((mapx >= 0) & (mapx < W) & (mapy >= 0) & (mapy < H))
    band = cv2.remap(g, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    # v방향(축) 그래디언트 = 링 에지 신호
    gv = np.abs(cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3))
    # 각 행에서 '전주 동시성': θ에 걸친 에지의 평균(강+고르게) / 유효 θ
    vok = valid.mean(1) > 0.7
    prof = np.where(valid, gv, 0.0).sum(1) / np.maximum(valid.sum(1), 1)
    prof[~vok] = 0.0
    if vok.sum() < NV * 0.4:
        rows.append((fi, 0.0, np.nan, 0.0))
        continue
    ri = int(np.argmax(prof))
    peak = float(prof[ri])
    med = float(np.median(prof[vok]))
    ratio = peak / max(med, 1e-6)          # 전주 링다움(주변 대비 돌출)
    rows.append((fi, peak, float(v[ri]), ratio))
    if fi % 5000 == 0:
        print(f"{fi}/{N}", flush=True)
cap.release()

arr = np.array(rows, dtype=np.float64)
np.savetxt(f"{OUT}/joint_{TAG}.csv", arr, delimiter=",",
           header="frame,peak,v_at_peak,ratio", comments="")
# 이벤트 후보: ratio가 국소 최대 + 임계 초과
r = arr[:, 3]
thr = np.nanpercentile(r, 90)
cand = []
for i in range(2, len(r) - 2):
    if r[i] >= thr and r[i] == max(r[i - 2:i + 3]):
        cand.append(int(arr[i, 0]))
# 근접 이벤트 병합(300프레임)
merged = []
for c in cand:
    if not merged or c - merged[-1] > 300:
        merged.append(c)
print(f"[{TAG}] frames={len(arr)} ratio p90={thr:.2f} "
      f"joint_candidates={len(merged)}: {merged[:30]}")
