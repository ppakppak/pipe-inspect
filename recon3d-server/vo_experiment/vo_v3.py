"""VO v3 — v2 + 적응형 밴드반경(관 벽 대역 자동검출).

진단(d200 부호반전): 관 벽이 화면상 차지하는 반경대가 구경·거리·구도로 크게
다름(fwd350 r45~90 vs d200 r100~190). 고정 RHO_IN이 소구경서 암부를 벽으로
오샘플 → 상관 오염·부호반전. 해결: 방사 밝기 프로파일로 벽 시작(급상승)~끝을
매 프레임 검출, 그 대역만 v=1/rho 샘플. EMA 안정화.

usage: vo_v3.py <video> <tag> [stop]
"""
import os
import sys
from collections import deque

import cv2
import numpy as np

V = sys.argv[1]; TAG = sys.argv[2]
STOP = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
OUT = "/tmp/vo_test"; os.makedirs(OUT, exist_ok=True)
STRIDE = 2; BASE = 12; NTH, NV = 192, 128; NSEC = 24
MAX_SHIFT = 24; MAX_ROLL = 24; DS = 0.5
OSD_RECTS = [(60, 130, 80, 500), (950, 1025, 1600, 1900)]

cap = cv2.VideoCapture(V)
N = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), STOP)
th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
cos_t, sin_t = np.cos(th), np.sin(th)
SPK = NTH // NSEC


def wall_band(g, cx, cy, RMAX):
    """방사 텍스처 에너지 프로파일 → 벽 대역 [r_lo, r_hi] 검출.

    벽=텍스처(그래디언트) 강한 반경대. 밝기는 반경 단조증가라 무용.
    극좌표 변환(warpPolar)으로 빠르게: 반경별 |grad| 평균 프로파일.
    """
    pol = cv2.warpPolar(g, (RMAX, 360), (cx, cy), RMAX,
                        cv2.INTER_LINEAR + cv2.WARP_POLAR_LINEAR)
    # pol: (360=angle, RMAX=radius). 반경축(가로) 그래디언트 = 축방향 텍스처
    gx = np.abs(cv2.Sobel(pol.astype(np.float32), cv2.CV_32F, 1, 0, 3))
    inb = pol > 3                         # 화면 밖(0패딩) 제외
    denom = np.maximum(inb.sum(0), 1)
    prof = (gx * inb).sum(0) / denom      # 반경별 평균 에너지
    cov = inb.mean(0)                     # 반경별 유효 각도 비율
    prof = prof * (cov > 0.3)             # 커버리지 낮은 반경 배제
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (0, 0), 4).ravel()
    if prof.max() < 1e-3:
        return None
    pk = int(np.argmax(prof))
    thr = 0.30 * prof[pk]                 # 피크의 30% 이상 = 벽 대역
    lo = pk
    while lo > 0 and prof[lo] >= thr:
        lo -= 1
    hi = pk
    while hi < RMAX - 1 and prof[hi] >= thr:
        hi += 1
    r_lo = max(lo, 12)
    r_hi = min(hi, RMAX - 1)
    if r_hi - r_lo < 20:
        return None
    return r_lo, r_hi


hist = deque(maxlen=BASE + 1)
center = None; band_r = None
rows = []; fi = -1
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
    # 반경 상한 = 어안 원형뷰 반경(비네팅 검은테 경계 배제). 편심은
    # warpPolar 0패딩+커버리지로 대응. hypot로 키우면 경계 에지가 벽을 이김.
    RMAX = int(0.92 * min(H, W) / 2)
    wb = wall_band(g, cx, cy, RMAX)
    if wb is None:
        rows.append((fi, np.nan, 0.0, 0, np.nan, 0.0, np.nan, np.nan))
        continue
    r_lo, r_hi = wb
    if band_r is None:
        band_r = np.array([r_lo, r_hi], float)
    else:
        band_r = 0.85 * band_r + 0.15 * np.array([r_lo, r_hi])  # EMA
    rl, rh = band_r
    v = np.linspace(1.0 / rh, 1.0 / rl, NV)      # 적응 대역
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
            aa, bb = (a[sh:], b[:NV - sh]) if sh >= 0 else (a[:NV + sh], b[-sh:])
            den = np.sqrt((aa * aa).sum() * (bb * bb).sum())
            if den < 1e-6:
                continue
            c = float((aa * bb).sum() / den)
            if c > best_c:
                best_c, best_s = c, sh
        if best_s is not None and best_c > 0.30:
            shifts.append(best_s); nccs.append(best_c)
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
            roll_deg, roll_c = best_r * 360.0 / NTH, best_rc
    if len(shifts) >= 4:
        dz_rows = float(np.median(shifts)); conf = float(np.median(nccs))
        dz_rel = -dz_rows * span / (NV - 1) / (BASE * STRIDE)
        rows.append((fi, dz_rel, conf, len(shifts), roll_deg, roll_c, rl, rh))
    else:
        rows.append((fi, np.nan, 0.0, len(shifts), roll_deg, roll_c, rl, rh))
    if fi % 5000 == 0:
        print(f"{fi}/{N} band=[{rl:.0f},{rh:.0f}]", flush=True)
cap.release()
arr = np.array(rows, dtype=np.float64)
np.savetxt(f"{OUT}/vo_v3_{TAG}.csv", arr, delimiter=",",
           header="frame,dz_rel,conf,nsec,roll_deg,roll_c,r_lo,r_hi", comments="")
ok = np.isfinite(arr[:, 1])
print(f"[{TAG}] pairs={len(arr)} valid={ok.sum()}({100*ok.mean():.0f}%) "
      f"conf_med={np.median(arr[ok,2]):.3f} dz>0={np.mean(arr[ok,1]>0):.2f} "
      f"band_med=[{np.nanmedian(arr[:,6]):.0f},{np.nanmedian(arr[:,7]):.0f}]")
