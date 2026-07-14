"""VO 품질 자기진단 — OSD 없이 영상별 유효/무효 판정.

지표:
- coverage: 유효 프레임 비율(밴드/중심 성립)
- straightness: 거시궤적 직진도 |net|/path (전진영상 단조성; U자반전=낮음)
- conf_med: 밴드상관 신뢰 중앙값
유효 = coverage>0.9 AND straightness>THR AND conf_med>0.4
무효 영상은 체인리지 '거리미상' 처리(본 만큼만 말한다).
"""
import numpy as np
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from vo_integrate import robust_cumulative

STRAIGHT_THR = 0.45


def straightness(cum, nb=40):
    idx = np.linspace(0, len(cum) - 1, nb + 1).astype(int)
    pts = cum[idx]
    net = abs(pts[-1] - pts[0]); path = np.abs(np.diff(pts)).sum()
    return net / max(path, 1e-9)


def assess(csv):
    a = np.genfromtxt(csv, delimiter=",", names=True)
    cov = float(np.mean(np.isfinite(a["dz_rel"])))
    conf = float(np.nanmedian(a["conf"][np.isfinite(a["dz_rel"])]))
    cum = robust_cumulative(a["frame"], a["dz_rel"], a["conf"])
    st = straightness(cum)
    ok = cov > 0.9 and st > STRAIGHT_THR and conf > 0.4
    return {"coverage": round(cov, 3), "straightness": round(st, 3),
            "conf_med": round(conf, 3), "valid": ok}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r = assess(p)
        tag = p.rsplit("/", 1)[-1]
        print(f"{tag}: valid={r['valid']} straight={r['straightness']} "
              f"cov={r['coverage']} conf={r['conf_med']}")
