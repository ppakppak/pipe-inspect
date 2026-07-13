"""시각 오도메트리 타당성 실험 v0 — 프레임 간 링밴드 위상상관 Δz 적분 vs OSD 주행거리계.

원리: 전방 카메라 + 원통 벽 ⇒ 벽점 z = fR/ρ. 밴드를 v=1/ρ 균일 샘플하면
전진 Δz가 v축 순수 평행이동(Δv = Δz/fR). 위상상관으로 (Δv, Δθ) 추정,
Δz는 상대단위로 적분 → OSD 체크포인트에 스케일 피팅해 비교.
"""
import os
import sys

import cv2
import numpy as np

V = sys.argv[1]
OUT = "/tmp/vo_test"
os.makedirs(OUT, exist_ok=True)

STRIDE = 2          # 2프레임 간격(12.5fps 유효)
DS = 0.5            # 다운스케일
NTH, NV = 256, 96   # 밴드 해상도(θ×v)
RHO_IN = 40.0       # 내측 반경(ds px) — 중심부(원거리·저해상)는 제외
METER_EVERY = 500   # OSD 미터 크롭 저장 간격(원본 프레임 번호 기준)
# OSD 영역(1920x1080 원본): 좌상 타임스탬프 / 우하 주행거리계
OSD_RECTS = [(60, 130, 80, 500), (950, 1025, 1600, 1900)]  # (y0,y1,x0,x1)
METER_CROP = (955, 1020, 1600, 1880)

cap = cv2.VideoCapture(V)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
cos_t, sin_t = np.cos(th), np.sin(th)
hann_v = np.hanning(NV)[:, None]

center = None
prev = None
rows = []          # fi, dv_rows, dth_cols, resp
fi = -1
while True:
    ret, fr = cap.read()
    if not ret:
        break
    fi += 1
    if fi % METER_EVERY == 0:
        y0, y1, x0, x1 = METER_CROP
        cv2.imwrite(f"{OUT}/meter_{fi:06d}.png", fr[y0:y1, x0:x1])
    if fi % STRIDE:
        continue
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    for (y0, y1, x0, x1) in OSD_RECTS:
        g[y0:y1, x0:x1] = int(g.mean())
    g = cv2.resize(g, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)
    H, W = g.shape
    # 중심 = 암부(보어) 무게중심, EMA 평활
    blur = cv2.GaussianBlur(g, (0, 0), 15)
    thr = np.percentile(blur, 5)
    ys, xs = np.nonzero(blur <= thr)
    if len(xs) < 50:
        rows.append((fi, np.nan, np.nan, 0.0))
        prev = None
        continue
    c_new = np.array([xs.mean(), ys.mean()])
    center = c_new if center is None else 0.9 * center + 0.1 * c_new
    cx, cy = center
    rho_out = min(cx, cy, W - 1 - cx, H - 1 - cy) * 0.95
    if rho_out < RHO_IN + 25:
        rows.append((fi, np.nan, np.nan, 0.0))
        prev = None
        continue
    v = np.linspace(1.0 / rho_out, 1.0 / RHO_IN, NV)
    rho = 1.0 / v
    mapx = (cx + rho[:, None] * cos_t[None, :]).astype(np.float32)
    mapy = (cy + rho[:, None] * sin_t[None, :]).astype(np.float32)
    band = cv2.remap(g, mapx, mapy, cv2.INTER_LINEAR).astype(np.float32)
    band -= cv2.GaussianBlur(band, (0, 0), 9)     # 저주파 조명 제거
    band *= hann_v                                 # v축 에지 테이퍼
    if prev is not None:
        (dx, dy), resp = cv2.phaseCorrelate(prev['band'], band)
        # dv를 Δz 상대단위로: 행간격 = (v_max-v_min)/(NV-1), Δz ∝ dv_rows·행간격
        dz_rel = dy * (prev['dvspan'] + (v[-1] - v[0])) * 0.5 / (NV - 1)
        rows.append((fi, dz_rel, dx * 360.0 / NTH, resp))
    prev = {'band': band, 'dvspan': v[-1] - v[0]}
    if fi % 5000 == 0:
        print(f"{fi}/{N}", flush=True)
cap.release()

arr = np.array(rows, dtype=np.float64)
np.savetxt(f"{OUT}/vo_raw.csv", arr, delimiter=",",
           header="frame,dz_rel,dth_deg,resp", comments="")
ok = np.isfinite(arr[:, 1])
print(f"pairs={len(arr)} valid={ok.sum()} resp_med={np.median(arr[ok, 3]):.3f}")
