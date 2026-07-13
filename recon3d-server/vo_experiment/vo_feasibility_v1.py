"""VO 타당성 v1 — v0 진단 반영: ①24프레임 베이스라인(시프트≫노이즈)
②섹터별 부분원호 1D 상관+중앙값 합의(사면 구도서 화면밖 섹터만 배제).
"""
import os
import sys
from collections import deque

import cv2
import numpy as np

V = sys.argv[1]
OUT = "/tmp/vo_test"
os.makedirs(OUT, exist_ok=True)

STRIDE = 2           # 처리 간격(원본 프레임)
BASE = 12            # 상관 베이스라인(처리 스텝) = 24 원본 프레임 ≈ 1초
NTH, NV = 192, 128   # 밴드 해상도(θ×v)
NSEC = 24            # 섹터 수(8스포크/섹터)
RHO_IN = 45.0        # 내측 반경(ds px)
MAX_SHIFT = 24       # 탐색 시프트(행)
DS = 0.5
OSD_RECTS = [(60, 130, 80, 500), (950, 1025, 1600, 1900)]

cap = cv2.VideoCapture(V)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
cos_t, sin_t = np.cos(th), np.sin(th)
SPK = NTH // NSEC

hist = deque(maxlen=BASE + 1)   # (band, valid, v0span)
center = None
rows = []
fi = -1
while True:
    ret, fr = cap.read()
    if not ret:
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
    thr = np.percentile(blur, 5)
    ys, xs = np.nonzero(blur <= thr)
    if len(xs) >= 50:
        c_new = np.array([xs.mean(), ys.mean()])
        center = c_new if center is None else 0.9 * center + 0.1 * c_new
    if center is None:
        center = np.array([W / 2.0, H / 2.0])
    cx, cy = center
    rho_out = 0.9 * min(H, W) / 2.0          # 중심 위치와 무관하게 고정 스팬
    v = np.linspace(1.0 / rho_out, 1.0 / RHO_IN, NV)
    rho = 1.0 / v
    mapx = (cx + rho[:, None] * cos_t[None, :]).astype(np.float32)
    mapy = (cy + rho[:, None] * sin_t[None, :]).astype(np.float32)
    valid = ((mapx >= 0) & (mapx < W) & (mapy >= 0) & (mapy < H))
    band = cv2.remap(g, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT).astype(np.float32)
    band -= cv2.GaussianBlur(band, (0, 0), 7)
    hist.append((band, valid, v[-1] - v[0]))
    if len(hist) <= BASE:
        continue
    b0, m0, span = hist[0]
    shifts, nccs = [], []
    for s in range(NSEC):
        sl = slice(s * SPK, (s + 1) * SPK)
        vs = m0[:, sl] & valid[:, sl]
        if vs.mean() < 0.7:
            continue
        w0 = np.where(vs, b0[:, sl], 0.0).sum(1) / np.maximum(vs.sum(1), 1)
        w1 = np.where(vs, band[:, sl], 0.0).sum(1) / np.maximum(vs.sum(1), 1)
        rowok = vs.mean(1) > 0.5
        if rowok.sum() < NV * 0.6:
            continue
        a = w0 - w0[rowok].mean()
        b = w1 - w1[rowok].mean()
        a[~rowok] = 0.0
        b[~rowok] = 0.0
        best_s, best_c = None, -1.0
        for sh in range(-MAX_SHIFT, MAX_SHIFT + 1):
            if sh >= 0:
                aa, bb = a[sh:], b[:NV - sh]
            else:
                aa, bb = a[:NV + sh], b[-sh:]
            den = np.sqrt((aa * aa).sum() * (bb * bb).sum())
            if den < 1e-6:
                continue
            c = float((aa * bb).sum() / den)
            if c > best_c:
                best_c, best_s = c, sh
        if best_s is not None and best_c > 0.30:
            shifts.append(best_s)
            nccs.append(best_c)
    if len(shifts) >= 4:
        dz_rows = float(np.median(shifts))
        conf = float(np.median(nccs))
        # 행→v→상대 Δz, 베이스라인(원본 프레임 수)으로 정규화
        dz_rel = dz_rows * span / (NV - 1) / (BASE * STRIDE)
        rows.append((fi, dz_rel, conf, len(shifts)))
    else:
        rows.append((fi, np.nan, 0.0, len(shifts)))
    if fi % 5000 == 0:
        print(f"{fi}/{N}", flush=True)
cap.release()

arr = np.array(rows, dtype=np.float64)
np.savetxt(f"{OUT}/vo_v1.csv", arr, delimiter=",",
           header="frame,dz_rel,conf,nsec", comments="")
ok = np.isfinite(arr[:, 1])
print(f"pairs={len(arr)} valid={ok.sum()} ({100 * ok.mean():.1f}%) "
      f"conf_med={np.median(arr[ok, 2]):.3f}")
