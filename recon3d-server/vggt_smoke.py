"""VGGT 관내부 스모크 테스트 — 관 내부 영상에서 3D(depth·포즈·포인트맵)가 성립하는가.

판정 기준:
  1) depth map이 원통형(중앙 멀고 가장자리 가까움)으로 나오는가
  2) 카메라 궤적이 관축 방향 직선인가
  3) 포인트클라우드 단면이 원/원호를 이루는가
사용: .venv/bin/python vggt_smoke.py <video> <center_frame> <out_prefix>
"""
import sys, os
import cv2
import numpy as np
import torch

VIDEO = sys.argv[1]
CENTER = int(sys.argv[2])
OUT = sys.argv[3]
N_FRAMES, STEP = 5, 12   # 8프레임 × 8프레임 간격

# ── 1) 프레임 추출 ──
cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
paths = []
os.makedirs("/tmp/vggt_frames", exist_ok=True)
for i in range(N_FRAMES):
    fi = min(max(CENTER + (i - N_FRAMES // 2) * STEP, 0), total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, fr = cap.read()
    if not ret:
        continue
    p = f"/tmp/vggt_frames/{OUT}_{i:02d}.jpg"
    cv2.imwrite(p, fr)
    paths.append(p)
cap.release()
print(f"frames: {len(paths)} (center {CENTER}, step {STEP})")

# ── 2) VGGT 추론 ──
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda"
dtype = torch.bfloat16
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
images = load_and_preprocess_images(paths).to(device)
with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
    pred = model(images)
print("pred keys:", list(pred.keys()))

extri, intri = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
extri = extri.squeeze(0).float().cpu().numpy()   # (S,3,4)
depth = pred["depth"].squeeze(0).float().cpu().numpy()       # (S,H,W,1)
conf = pred["depth_conf"].squeeze(0).float().cpu().numpy()   # (S,H,W)
wp = pred["world_points"].squeeze(0).float().cpu().numpy()   # (S,H,W,3)
S, H, W = depth.shape[0], depth.shape[1], depth.shape[2]
print(f"depth {depth.shape}  conf mean {conf.mean():.2f}  world_points {wp.shape}")

# ── 3) depth 컬러맵 저장 (첫/중간 프레임) ──
for i in (0, S // 2):
    d = depth[i, :, :, 0]
    dn = (d - np.percentile(d, 2)) / max(np.percentile(d, 98) - np.percentile(d, 2), 1e-6)
    dn = np.clip(dn, 0, 1)
    cm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    orig = cv2.resize(cv2.imread(paths[i]), (W, H))
    cv2.imwrite(f"/tmp/{OUT}_depth{i}.jpg", np.hstack([orig, cm]))

# ── 4) 카메라 궤적 ──
# extrinsic = world→cam [R|t] → cam center = -R^T t
centers = np.array([-e[:3, :3].T @ e[:3, 3] for e in extri])
seg = centers[-1] - centers[0]
seglen = np.linalg.norm(seg)
lin = np.linalg.norm(centers - (centers[0] + np.outer(
    np.dot(centers - centers[0], seg / max(seglen, 1e-9)), seg / max(seglen, 1e-9))), axis=1)
print("camera centers (VGGT scale):")
for c in centers:
    print(f"  ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})")
print(f"궤적 길이={seglen:.4f}, 직선 이탈 max={lin.max():.4f} ({lin.max()/max(seglen,1e-9)*100:.1f}%)")

# ── 5) 포인트클라우드 단면/측면 뷰 ──
i = S // 2
pts = wp[i].reshape(-1, 3)
cf = conf[i].reshape(-1)
m = cf > np.percentile(cf, 50)   # 상위 50% 신뢰
pts = pts[m]
# 카메라 진행축 = seg 방향 → 단면은 그 직교평면
ax = seg / max(seglen, 1e-9)
tmp = np.array([1.0, 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1.0, 0])
u = np.cross(ax, tmp); u /= np.linalg.norm(u)
v = np.cross(ax, u)
pu, pv, pz = pts @ u, pts @ v, pts @ ax


def scatter_img(x, y, w=800, h=800, name="view"):
    img = np.zeros((h, w, 3), np.uint8)
    x1, x2 = np.percentile(x, 1), np.percentile(x, 99)
    y1, y2 = np.percentile(y, 1), np.percentile(y, 99)
    r = max(x2 - x1, y2 - y1, 1e-9)
    xi = ((x - x1) / r * (w - 20) + 10).astype(int)
    yi = ((y - y1) / r * (h - 20) + 10).astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    img[yi[ok], xi[ok]] = (80, 220, 80)
    cv2.imwrite(f"/tmp/{OUT}_{name}.jpg", img)


scatter_img(pu, pv, name="crosssec")   # 단면: 원이어야 함
scatter_img(pz, pv, name="side")       # 측면: 두 평행선(관벽)이어야 함
print(f"saved: /tmp/{OUT}_depth*.jpg /tmp/{OUT}_crosssec.jpg /tmp/{OUT}_side.jpg")
