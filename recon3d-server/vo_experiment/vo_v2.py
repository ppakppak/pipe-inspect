"""VO v2 — v1(섹터 축방향 합의) + 롤(원주 회전) 추정 + 부호 명확화.

밴드(θ×v)에서 축전진=v축 이동, 롤=θ축 순환 이동 — 직교하므로 분리 1D 상관.
축 Δz: 섹터별 v-시프트 median 합의(v1). 부호 전진=양수(-median(shift)).
롤 Δθ: v방향 평균 θ-프로파일의 순환 상관 argmax.
정지=시프트 0 자연 측정, 후진=음수.

usage: vo_v2.py <video> <out_tag> [stop_frame]
"""
import os
import sys
from collections import deque

import cv2
import numpy as np

V = sys.argv[1]
TAG = sys.argv[2]
STOP = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
OUT = "/tmp/vo_test"
os.makedirs(OUT, exist_ok=True)

STRIDE = 2
BASE = 12
NTH, NV = 192, 128
NSEC = 24
RHO_IN = 45.0
MAX_SHIFT = 24
MAX_ROLL = 24       # θ 시프트 탐색(스포크)
DS = 0.5
OSD_RECTS = [(60, 130, 80, 500), (950, 1025, 1600, 1900)]

cap = cv2.VideoCapture(V)
N = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), STOP)
th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
cos_t, sin_t = np.cos(th), np.sin(th)
SPK = NTH // NSEC
hann_th = np.hanning(NV)[:, None]

hist = deque(maxlen=BASE + 1)
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
    thr = np.percentile(blur, 5)
    ys, xs = np.nonzero(blur <= thr)
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
    band -= cv2.GaussianBlur(band, (0, 0), 7)
    hist.append((band, valid, v[-1] - v[0]))
    if len(hist) <= BASE:
        continue
    b0, m0, span = hist[0]
    both = m0 & valid
    # ── 축방향 Δz: 섹터별 v-시프트 median 합의 ──
    shifts, nccs = [], []
    for s in range(NSEC):
        sl = slice(s * SPK, (s + 1) * SPK)
        vs = both[:, sl]
        if vs.mean() < 0.7:
            continue
        w0 = np.where(vs, b0[:, sl], 0.0).sum(1) / np.maximum(vs.sum(1), 1)
        w1 = np.where(vs, band[:, sl], 0.0).sum(1) / np.maximum(vs.sum(1), 1)
        rowok = vs.mean(1) > 0.5
        if rowok.sum() < NV * 0.6:
            continue
        a = w0 - w0[rowok].mean(); b = w1 - w1[rowok].mean()
        a[~rowok] = 0.0; b[~rowok] = 0.0
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
            shifts.append(best_s); nccs.append(best_c)
    # ── 롤 Δθ: θ-프로파일 순환 상관(전 밴드 v평균) ──
    roll_deg, roll_c = np.nan, 0.0
    colok = both.mean(0) > 0.6
    if colok.sum() > NTH * 0.5:
        p0 = np.where(both, b0, 0.0).sum(0) / np.maximum(both.sum(0), 1)
        p1 = np.where(both, band, 0.0).sum(0) / np.maximum(both.sum(0), 1)
        p0 = p0 - p0[colok].mean(); p1 = p1 - p1[colok].mean()
        p0[~colok] = 0.0; p1[~colok] = 0.0
        best_r, best_rc = 0, -1.0
        for rsh in range(-MAX_ROLL, MAX_ROLL + 1):
            bb = np.roll(p1, rsh)
            den = np.sqrt((p0 * p0).sum() * (bb * bb).sum())
            if den < 1e-6:
                continue
            c = float((p0 * bb).sum() / den)
            if c > best_rc:
                best_rc, best_r = c, rsh
        if best_rc > 0.30:
            roll_deg = best_r * 360.0 / NTH
            roll_c = best_rc
    if len(shifts) >= 4:
        dz_rows = float(np.median(shifts))
        conf = float(np.median(nccs))
        # 전진=양수: 전진 시 텍스처가 바깥(행번호↓)으로 흘러 shift<0
        dz_rel = -dz_rows * span / (NV - 1) / (BASE * STRIDE)
        rows.append((fi, dz_rel, conf, len(shifts), roll_deg, roll_c))
    else:
        rows.append((fi, np.nan, 0.0, len(shifts), roll_deg, roll_c))
    if fi % 5000 == 0:
        print(f"{fi}/{N}", flush=True)
cap.release()

arr = np.array(rows, dtype=np.float64)
np.savetxt(f"{OUT}/vo_v2_{TAG}.csv", arr, delimiter=",",
           header="frame,dz_rel,conf,nsec,roll_deg,roll_c", comments="")
ok = np.isfinite(arr[:, 1])
cum = np.nansum(np.where(ok, arr[:, 1], 0.0))
rok = np.isfinite(arr[:, 4])
roll_cum = np.nansum(np.where(rok, arr[:, 4], 0.0)) if rok.any() else 0.0
print(f"[{TAG}] pairs={len(arr)} valid={ok.sum()}({100*ok.mean():.0f}%) "
      f"conf_med={np.median(arr[ok,2]):.3f} dz_sum={cum:+.1f}(rel) "
      f"roll_valid={rok.sum()} roll_cum={roll_cum:+.0f}deg")
