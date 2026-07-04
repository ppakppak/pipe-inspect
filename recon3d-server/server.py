"""3D 복원(recon3d) 마이크로서비스 — VGGT 기반 관내부 3D 스모크/실험.

포트 5006, 127.0.0.1 전용(프록시 5003 경유 접근). 전용 venv(.venv)로
프로덕션 gpu-server(5004) 무오염. VRAM 예산이 빠듯해(5004가 ~13GB 상주)
프레임 수 상한 6, OOM 시 사용자 친화 에러 반환.

증분 1: 프레임 구간 → VGGT → depth 컬러맵·포인트클라우드 단면·궤적·conf 판정.
이후 증분: 원통 피팅(관경 앵커) → 전개 → 면적.
"""
import base64
import os
import time
import uuid

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel

ALLOWED_ROOTS = [
    os.environ.get('KWATER_VIDEO_ROOT', '/home/intu/nas2/k_water/관내시경영상'),
    '/home/intu/projects/pipe-field-nex/test_videos',
]
MAX_FRAMES = 6

app = FastAPI(title="recon3d (VGGT)")
_model = None


def get_model():
    global _model
    if _model is None:
        from vggt.models.vggt import VGGT
        _model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
    return _model


def b64jpg(img, q=85):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()


class RunReq(BaseModel):
    video_path: str
    center_frame: int = 0
    n_frames: int = 5
    step: int = 12
    clahe: bool = False


@app.get('/health')
def health():
    free, total = torch.cuda.mem_get_info()
    return {'success': True, 'model_loaded': _model is not None,
            'vram_free_gb': round(free / 2 ** 30, 2),
            'vram_total_gb': round(total / 2 ** 30, 2),
            'max_frames': MAX_FRAMES}


@app.get('/video-info')
def video_info(path: str):
    p = os.path.realpath(path)
    if not any(p.startswith(os.path.realpath(r) + os.sep) for r in ALLOWED_ROOTS):
        return {'success': False, 'error': '허용 경로 밖'}
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        return {'success': False, 'error': '영상 열기 실패'}
    info = {'success': True,
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': round(cap.get(cv2.CAP_PROP_FPS) or 0, 1),
            'width': int(cap.get(3)), 'height': int(cap.get(4))}
    cap.release()
    return info


@app.post('/run')
def run(req: RunReq):
    p = os.path.realpath(req.video_path)
    if not any(p.startswith(os.path.realpath(r) + os.sep) for r in ALLOWED_ROOTS):
        return {'success': False, 'error': '허용 경로 밖 파일'}
    if not os.path.isfile(p):
        return {'success': False, 'error': f'파일 없음: {p}'}

    n = max(2, min(int(req.n_frames), MAX_FRAMES))
    step = max(1, int(req.step))

    # ── 프레임 추출 ──
    cap = cv2.VideoCapture(p)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    c = max(0, min(int(req.center_frame), max(total - 1, 0)))
    clahe = cv2.createCLAHE(2.0, (8, 8)) if req.clahe else None
    tmpd = f"/tmp/recon3d_{uuid.uuid4().hex[:8]}"
    os.makedirs(tmpd, exist_ok=True)
    paths, used = [], []
    for i in range(n):
        fi = min(max(c + (i - n // 2) * step, 0), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, fr = cap.read()
        if not ret:
            continue
        if clahe is not None:
            lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            fr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        fp = f"{tmpd}/{i:02d}.jpg"
        cv2.imwrite(fp, fr)
        paths.append(fp)
        used.append(int(fi))
    cap.release()
    if len(paths) < 2:
        return {'success': False, 'error': '프레임 추출 실패'}

    # ── VGGT 추론 ──
    try:
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        model = get_model()
        images = load_and_preprocess_images(paths).to("cuda")
        t0 = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = model(images)
        infer_s = time.time() - t0
        extri, _ = pose_encoding_to_extri_intri(pred['pose_enc'], images.shape[-2:])
        extri = extri.squeeze(0).float().cpu().numpy()
        depth = pred['depth'].squeeze(0).float().cpu().numpy()
        conf = pred['depth_conf'].squeeze(0).float().cpu().numpy()
        wp = pred['world_points'].squeeze(0).float().cpu().numpy()
        del pred, images
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {'success': False,
                'error': f'VRAM 부족 — 프레임 수를 줄이세요 (현재 {n}). '
                         f'5004 GPU서버가 VRAM을 점유 중일 수 있습니다.'}

    S, H, W = depth.shape[0], depth.shape[1], depth.shape[2]
    mid = S // 2

    # ── depth 컬러맵 (원본 | depth) ──
    d = depth[mid, :, :, 0]
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    dn = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    cm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    orig = cv2.resize(cv2.imread(paths[mid]), (W, H))
    depth_img = np.hstack([orig, cm])

    # ── 카메라 궤적 ──
    centers = np.array([-e[:3, :3].T @ e[:3, 3] for e in extri])
    seg = centers[-1] - centers[0]
    seglen = float(np.linalg.norm(seg))
    axd = seg / max(seglen, 1e-9)
    proj = (centers - centers[0]) @ axd
    lin = np.linalg.norm((centers - centers[0]) - np.outer(proj, axd), axis=1)
    traj_dev_pct = float(lin.max() / max(seglen, 1e-9) * 100)

    # ── 포인트클라우드 단면 산점도 ──
    tmpv = np.array([1., 0, 0]) if abs(axd[0]) < 0.9 else np.array([0, 1., 0])
    u = np.cross(axd, tmpv); u /= np.linalg.norm(u)
    v = np.cross(axd, u)
    pts = wp[mid].reshape(-1, 3)
    cf = conf[mid].reshape(-1)
    pts = pts[cf > np.percentile(cf, 50)]
    pu, pv = pts @ u, pts @ v
    cs = np.zeros((640, 640, 3), np.uint8)
    x1, x2 = np.percentile(pu, 1), np.percentile(pu, 99)
    y1, y2 = np.percentile(pv, 1), np.percentile(pv, 99)
    r = max(x2 - x1, y2 - y1, 1e-9)
    xi = ((pu - x1) / r * 620 + 10).astype(int)
    yi = ((pv - y1) / r * 620 + 10).astype(int)
    ok = (xi >= 0) & (xi < 640) & (yi >= 0) & (yi < 640)
    cs[yi[ok], xi[ok]] = (80, 220, 80)

    cmean = float(conf.mean())
    verdict = ('성립 (원통 피팅 가능성 높음)' if cmean > 2.0
               else '불확실 — 구간/베이스라인/CLAHE 바꿔 재시도' if cmean > 1.3
               else '실패 가능성 높음 (장면 정보 빈곤)')
    return {'success': True, 'conf_mean': round(cmean, 2), 'verdict': verdict,
            'infer_sec': round(infer_s, 1), 'n_frames': S, 'frames_used': used,
            'traj_len': round(seglen, 4), 'traj_dev_pct': round(traj_dev_pct, 1),
            'depth_b64': b64jpg(depth_img), 'crosssec_b64': b64jpg(cs)}


class ScanReq(BaseModel):
    video_path: str
    start_frame: int = 0
    end_frame: int = 0        # 0 = 끝까지
    n_points: int = 40
    n_frames: int = 2         # 포인트당 프레임(저비용 스캔용 2)
    step: int = 12
    clahe: bool = False


@app.post('/scan')
def scan(req: ScanReq):
    """영상 전체를 균등 스윕해 conf 프로파일(3D 성립 구간 지도) 생성.

    포인트당 n_frames(기본 2)로 가볍게 VGGT를 돌려 depth_conf 평균만 수집.
    40포인트 기준 ~30초. 성립 섬(island)을 찾은 뒤 /run으로 정밀 실행.
    """
    p = os.path.realpath(req.video_path)
    if not any(p.startswith(os.path.realpath(r) + os.sep) for r in ALLOWED_ROOTS):
        return {'success': False, 'error': '허용 경로 밖 파일'}
    if not os.path.isfile(p):
        return {'success': False, 'error': f'파일 없음: {p}'}

    cap = cv2.VideoCapture(p)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = max(0, min(int(req.start_frame), max(total - 1, 0)))
    e = int(req.end_frame) if req.end_frame and int(req.end_frame) > s else total - 1
    e = min(e, total - 1)
    npts = max(5, min(int(req.n_points), 100))
    nfr = max(2, min(int(req.n_frames), 4))
    step = max(1, int(req.step))
    centers = np.linspace(s, e, npts).astype(int)
    clahe_op = cv2.createCLAHE(2.0, (8, 8)) if req.clahe else None

    from vggt.utils.load_fn import load_and_preprocess_images
    model = get_model()
    tmpd = f"/tmp/recon3d_scan_{uuid.uuid4().hex[:8]}"
    os.makedirs(tmpd, exist_ok=True)
    points = []
    t0 = time.time()
    for ci, c in enumerate(centers):
        paths = []
        for i in range(nfr):
            fi = min(max(int(c) + (i - nfr // 2) * step, 0), total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, fr = cap.read()
            if not ret:
                continue
            if clahe_op is not None:
                lab = cv2.cvtColor(fr, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe_op.apply(lab[:, :, 0])
                fr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            fp = f"{tmpd}/{ci:03d}_{i}.jpg"
            cv2.imwrite(fp, fr)
            paths.append(fp)
        if len(paths) < 2:
            points.append({'frame': int(c), 'conf': None})
            continue
        try:
            images = load_and_preprocess_images(paths).to("cuda")
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(images)
            cmean = float(pred['depth_conf'].float().mean())
            del pred, images
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            points.append({'frame': int(c), 'conf': None})
            continue
        if ci % 8 == 7:
            torch.cuda.empty_cache()
        points.append({'frame': int(c), 'conf': round(cmean, 2)})
    cap.release()
    torch.cuda.empty_cache()

    valid = [pt for pt in points if pt['conf'] is not None]
    best = sorted(valid, key=lambda x: -x['conf'])[:5]
    return {'success': True, 'total_frames': total, 'points': points,
            'best': best, 'scan_sec': round(time.time() - t0, 1),
            'n_frames': nfr, 'step': step}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5006)
