"""VO 적분 vs OSD GT 비교 분석. GT는 미터 크롭 육안 판독값(frame,meters) CSV."""
import sys

import numpy as np

raw = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
gt = np.genfromtxt(sys.argv[2], delimiter=",", names=True)  # frame,meters

f = raw["frame"]
dz = raw["dz_rel"]
resp = raw["resp"]
ok = np.isfinite(dz)
print(f"pairs={len(f)} valid={ok.sum()} ({100*ok.mean():.1f}%) resp: "
      f"med={np.median(resp[ok]):.3f} p10={np.percentile(resp[ok],10):.3f}")

# 신뢰 게이트: resp 하위는 0 처리(홀드)
for gate in [0.0, 0.1, 0.2, 0.3]:
    d = np.where(ok & (resp >= gate), np.nan_to_num(dz), 0.0)
    cum = np.cumsum(d)
    # GT 프레임 위치의 누적값 보간
    ci = np.interp(gt["frame"], f, cum)
    g = gt["meters"] - gt["meters"][0]
    # 최소자승 스케일(원점 통과): k = Σ(ci·g)/Σ(ci²)
    k = np.sum(ci * g) / max(np.sum(ci * ci), 1e-12)
    pred = k * ci
    r = np.corrcoef(ci, g)[0, 1]
    rmse = float(np.sqrt(np.mean((pred - g) ** 2)))
    # 구간별(체크포인트 간) 부호 일치율
    dg = np.diff(g)
    dc = np.diff(ci)
    sign_ok = float(np.mean(np.sign(dg[dg != 0]) == np.sign(dc[dg != 0]))) if (dg != 0).any() else float("nan")
    print(f"gate resp>={gate}: pearson r={r:.4f} scale k={k:.4g} "
          f"RMSE={rmse:.2f}m (GT range {g.max():.1f}m) 구간부호일치={sign_ok:.2f}")

# 최종 곡선 저장(플롯용)
d = np.where(ok, np.nan_to_num(dz), 0.0)
cum = np.cumsum(d)
np.savetxt("/tmp/vo_test/vo_cum.csv",
           np.column_stack([f, cum]), delimiter=",",
           header="frame,cum_rel", comments="")
