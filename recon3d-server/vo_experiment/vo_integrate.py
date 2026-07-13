"""VO CSV → 로버스트 체인리지 적분(공용). conf 게이트 + MAD 클립.

전진=양·후진=음 부호 분리, 무텍스처/암부 스파이크 억제. 스케일 상수 k는
OSD/조인트/VGGT 교정으로 결정(여기선 상대 적분까지).
"""
import numpy as np


def robust_cumulative(frame, dz_rel, conf, gate=0.6, mad_k=6.0):
    ok = np.isfinite(dz_rel) & (conf >= gate)
    d = np.where(ok, dz_rel, 0.0)
    v = d[ok]
    if len(v) == 0:
        return np.zeros_like(d)
    med = np.median(v)
    mad = np.median(np.abs(v - med)) + 1e-9
    d = np.clip(d, med - mad_k * mad, med + mad_k * mad)
    return np.cumsum(d)


def load_and_integrate(csv_path, **kw):
    a = np.genfromtxt(csv_path, delimiter=",", names=True)
    cum = robust_cumulative(a["frame"], a["dz_rel"], a["conf"], **kw)
    return a["frame"], cum, a
