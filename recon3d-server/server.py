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
    _model.to("cuda")   # 유휴 시 CPU 상주 → 사용 시 복귀(~1s)
    return _model


def release_model():
    """요청 종료 후 가중치를 CPU로 — 동거 GPU 프로세스(5004/8085)에 VRAM 반납."""
    global _model
    if _model is not None:
        _model.to("cpu")
        torch.cuda.empty_cache()


def b64jpg(img, q=85):
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()


def _brightness_corr(depth_mid, frame_path):
    """밝기↔depth 상관(OSD 상하 12/10% 제외) — 반전 감지 시그니처.

    관내부=손전등 물리(밝음=가까움)라 정상 프레임은 강한 음수(≈-0.7).
    양수(>+0.25)면 모델이 '밝음=멀다' 일반사진 prior로 폴백한 반전 의심.
    """
    g = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    H, W = depth_mid.shape
    g = cv2.resize(g, (W, H)).astype(np.float32)
    sl = slice(int(H * 0.12), int(H * 0.9))
    try:
        return float(np.corrcoef(g[sl].ravel(), depth_mid[sl].ravel())[0, 1])
    except Exception:
        return None


def _photometric_panel(frame_path, W, H):
    """photometric depth 컬러맵(1/√조명성분) — 반전 프레임 보조 표시용."""
    g = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    g = cv2.resize(g, (W, H)).astype(np.float32)
    I = cv2.GaussianBlur(g, (0, 0), 21)
    dp = 1.0 / np.sqrt(np.clip(I, 2, None))
    lo, hi = np.percentile(dp, 2), np.percentile(dp, 98)
    dn = np.clip((dp - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


INV_CORR_THR = 0.25

_yolo = None
DEFECT_MODEL = os.environ.get(
    "DEFECT_MODEL",
    "/home/intu/projects/pipe-field-nex/models/pipe_nodule_peel_2class_img960.pt")
DEFECT_NAMES_KO = {"corrosion_nodule": "결절", "coating_peel": "박리"}
DEFECT_COLORS = {"corrosion_nodule": (0, 0, 255), "coating_peel": (0, 200, 0)}
DEFECT_LABEL_EN = {"corrosion_nodule": "NODULE", "coating_peel": "PEEL"}  # cv2 putText는 ASCII만


def _get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(DEFECT_MODEL)
    return _yolo


class RunReq(BaseModel):
    video_path: str
    center_frame: int = 0
    n_frames: int = 5
    step: int = 12
    clahe: bool = False
    debug_depth: bool = False   # depth/conf/gray 원본 배열(npz) 반환 — 분석용
    diameter_mm: int = 0        # >0이면 부분원호 원통 피팅+metric 전개 수행
    measure_defects: bool = True  # 피팅 시 YOLO 결함 마스크를 표면 투영해 실면적(mm²) 산출
    det_conf: float = 0.30      # 박리 채택 임계(1차 검출은 내부 0.10 느슨)
    det_conf_nodule: float = 0.20  # 결절 채택 임계(성능 비대칭 반영해 낮춤)
    min_hits: int = 2           # 다중 프레임 지지 최소 수(2-of-N)
    detect_on_unwrap: bool = False  # (실험) 전개도 2차 검출


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
    try:
        return _run_impl(req)
    finally:
        release_model()   # 유휴 VRAM 반납


def _run_impl(req: RunReq):
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
        extri, intri = pose_encoding_to_extri_intri(pred['pose_enc'], images.shape[-2:])
        extri = extri.squeeze(0).float().cpu().numpy()
        intri = intri.squeeze(0).float().cpu().numpy()
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
    selm = cf > np.percentile(cf, 50)
    pts = pts[selm]
    pu, pv = pts @ u, pts @ v
    cs = np.zeros((640, 640, 3), np.uint8)
    x1, x2 = np.percentile(pu, 1), np.percentile(pu, 99)
    y1, y2 = np.percentile(pv, 1), np.percentile(pv, 99)
    r = max(x2 - x1, y2 - y1, 1e-9)
    xi = ((pu - x1) / r * 620 + 10).astype(int)
    yi = ((pv - y1) / r * 620 + 10).astype(int)
    ok = (xi >= 0) & (xi < 640) & (yi >= 0) & (yi < 640)
    cs[yi[ok], xi[ok]] = (150, 150, 150)

    cmean = float(conf.mean())
    cam_fwd = extri[mid][:3, :3].T @ np.array([0., 0., 1.])
    _lap, _bore = _frame_stats(paths[mid])
    met = _pipe_metrics(wp[mid], conf[mid], cam_forward=cam_fwd, lapvar=_lap, bore_frac=_bore)
    # 반전 감지: 밝기↔depth 상관 양수 = 조명 prior 폴백 의심 → 감점 + photometric 패널
    photo_corr = _brightness_corr(depth[mid, :, :, 0], paths[mid])
    depth_inverted = photo_corr is not None and photo_corr > INV_CORR_THR
    if depth_inverted:
        met['pipe_score'] = min(met['pipe_score'], 10)
        pm = _photometric_panel(paths[mid], W, H)
        if pm is not None:
            depth_img = np.hstack([depth_img, pm])
    score = met['pipe_score']
    # 판정은 pipe_score 우선 (conf 단독은 탁수/벽클로즈업을 고평가하는 함정)
    if depth_inverted:
        verdict = ('실패 — ⚠️depth 방향 반전 의심(밝기상관 %+.2f, '
                   '3번째 패널 photometric 참조)' % photo_corr)
    else:
        verdict = ('성립 — 관벽 링 구조 확인 (원통 피팅 가능)' if score >= 50
               else '불확실 — 링 부분 검출. 구간/베이스라인/CLAHE 재시도' if score >= 25
               else '실패 — 관벽 미노출(탁수·부유물·벽 클로즈업·블러 가능성)')
    resp_extra = {}
    viz_masks = []
    if req.diameter_mm and req.diameter_mm > 0 and not depth_inverted:
        try:
            cam_c = -extri[mid][:3, :3].T @ extri[mid][:3, 3]
            ret = _fit_cylinder_partial(wp, conf, paths, mid,
                                        cam_fwd, int(req.diameter_mm),
                                        cam_center=cam_c,
                                        measure_defects=req.measure_defects,
                                        K=intri[mid], extri_mid=extri[mid],
                                        extri_all=extri,
                                        det_conf=req.det_conf,
                                        det_conf_nodule=req.det_conf_nodule,
                                        min_hits=req.min_hits,
                                        detect_on_unwrap=req.detect_on_unwrap)
            cyl, viz_masks, wall_flat = ret if ret else (None, [], None)
            resp_extra['cyl_fit'] = cyl if cyl else {'error': '피팅 실패(포인트 부족)'}
            # 결함 위치를 원본 패널·단면 산점도에도 표시
            left = np.ascontiguousarray(depth_img[:, :W])
            for cls_raw, mimg in viz_masks:
                col = DEFECT_COLORS.get(cls_raw, (0, 255, 255))
                cts, _ = cv2.findContours(mimg.astype(np.uint8),
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(left, cts, -1, col, 2)
                dfull = mimg.reshape(-1)
                if wall_flat is not None:
                    dfull = dfull & wall_flat   # 측정에 든 점(벽면 인라이어)만 색칠
                df = dfull[selm]
                sel2 = ok & df
                cs[yi[sel2], xi[sel2]] = col
            depth_img[:, :W] = left
            if viz_masks:
                cv2.putText(cs, "NODULE", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, DEFECT_COLORS["corrosion_nodule"], 2)
                cv2.putText(cs, "PEEL", (95, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, DEFECT_COLORS["coating_peel"], 2)
        except Exception as e:
            resp_extra['cyl_fit'] = {'error': f'피팅 예외: {e}'}
    # ── 회전 뷰어용 서브샘플 클라우드 (int16 양자화 + RGB, 결함 클래스색) ──
    try:
        flat_idx = np.where(selm)[0]
        if len(flat_idx) > 45000:
            flat_idx = flat_idx[np.linspace(0, len(flat_idx) - 1, 45000).astype(int)]
        P3 = wp[mid].reshape(-1, 3)[flat_idx].astype(np.float32)
        colb = orig.reshape(-1, 3)[flat_idx][:, ::-1].copy()   # BGR→RGB
        for cls_raw, mimg in viz_masks:
            dfull = mimg.reshape(-1)
            if wall_flat is not None:
                dfull = dfull & wall_flat   # 칠해진 점 = 면적 계산에 든 점
            dmm = dfull[flat_idx]
            bgr = DEFECT_COLORS.get(cls_raw, (0, 255, 255))
            colb[dmm] = bgr[::-1]
        offp = P3.mean(0)
        scp = float(np.abs(P3 - offp).max()) / 32000.0 or 1.0
        xyz16 = np.clip((P3 - offp) / scp, -32700, 32700).astype(np.int16)
        resp_extra['cloud'] = {
            'n': int(len(flat_idx)),
            'xyz_b64': base64.b64encode(xyz16.tobytes()).decode(),
            'rgb_b64': base64.b64encode(np.ascontiguousarray(colb).tobytes()).decode()}
    except Exception:
        pass
    if req.debug_depth:
        import io as _io
        buf = _io.BytesIO()
        gray_mid = cv2.cvtColor(cv2.imread(paths[mid]), cv2.COLOR_BGR2GRAY)
        np.savez_compressed(buf, depth=depth[mid, :, :, 0].astype(np.float32),
                            conf=conf[mid].astype(np.float32),
                            gray=cv2.resize(gray_mid, (W, H)))
        resp_extra['debug_npz_b64'] = base64.b64encode(buf.getvalue()).decode()
    return {**resp_extra, 'success': True, 'conf_mean': round(cmean, 2), 'verdict': verdict,
            'pipe_score': score, 'cyl_res': met['cyl_res'], 'ang_cov': met['ang_cov'], 'axis': met.get('axis'), 'slab_ratio': met.get('slab_ratio'), 'lapvar': met.get('lapvar'), 'bore_frac': met.get('bore_frac'), 'photo_corr': None if photo_corr is None else round(photo_corr, 3), 'depth_inverted': bool(depth_inverted),
            'infer_sec': round(infer_s, 1), 'n_frames': S, 'frames_used': used,
            'traj_len': round(seglen, 4), 'traj_dev_pct': round(traj_dev_pct, 1),
            'depth_b64': b64jpg(depth_img), 'crosssec_b64': b64jpg(cs)}


def _ring_fit(pts, ax):
    """주어진 축으로 직교평면 투영 → Kasa 원피팅(아웃라이어 1회 제거 재적합).
    반환: (rel_res, ang_cov, inlier_frac) 또는 None."""
    q = pts - pts.mean(0)
    tmp = np.array([1., 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1., 0])
    u = np.cross(ax, tmp); u /= np.linalg.norm(u)
    vv = np.cross(ax, u)
    x, y = q @ u, q @ vv
    def kasa(x, y):
        A = np.stack([x, y, np.ones_like(x)], 1)
        b = -(x ** 2 + y ** 2)
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = -sol[0] / 2, -sol[1] / 2
        R2 = cx * cx + cy * cy - sol[2]
        return (cx, cy, float(np.sqrt(R2))) if R2 > 0 else None
    try:
        f = kasa(x, y)
        if f is None:
            return None
        cx, cy, R = f
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        inl = np.abs(r - R) < 0.3 * R
        inlier_frac = float(inl.mean())
        if inl.sum() > 300:
            f2 = kasa(x[inl], y[inl])
            if f2 is not None:
                cx, cy, R = f2
                r_in = np.sqrt((x[inl] - cx) ** 2 + (y[inl] - cy) ** 2)
            else:
                r_in = r[inl]
        else:
            r_in = r[inl] if inl.any() else r
        rel_res = float(np.sqrt(np.mean((r_in - R) ** 2)) / max(R, 1e-9))
        ang = np.arctan2(y[inl] - cy, x[inl] - cx) if inl.any() else np.arctan2(y - cy, x - cx)
        bins = np.unique(((ang + np.pi) / (2 * np.pi) * 36).astype(int) % 36)
        # 축방향 2슬랩 반경 일관성 — 원통=1.0, 안개 원뿔/그릇=낮음
        slab_ratio = 0.0
        t = q @ ax
        if inl.sum() > 600:
            tm = np.median(t[inl])
            Ra = Rb = None
            for half in (inl & (t <= tm), inl & (t > tm)):
                if half.sum() > 300:
                    fh = kasa(x[half], y[half])
                    if fh is not None:
                        if Ra is None:
                            Ra = fh[2]
                        else:
                            Rb = fh[2]
            if Ra and Rb:
                slab_ratio = float(min(Ra, Rb) / max(Ra, Rb))
        return rel_res, float(len(bins) / 36.0), inlier_frac, slab_ratio
    except np.linalg.LinAlgError:
        return None


def _frame_stats(path):
    """프레임 신호: (lapvar, bore_frac). bore_frac=암부(<60) 비율(OSD 상하 10% 제외).
    열린 보어 뷰=원거리 암흑 구멍 존재(수십%), 안개/탁수=산란광으로 전체가 밝아 ~0%."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    h = int(img.shape[0] * 640 / img.shape[1])
    img = cv2.resize(img, (640, h))
    lap = float(cv2.Laplacian(img, cv2.CV_64F).var())
    core = img[int(h * 0.1):int(h * 0.9)]
    bore = float((core < 60).mean())
    return lap, bore


def _pipe_metrics(wp, conf, cam_forward=None, lapvar=None, bore_frac=None):
    """'관다움' 측정 — conf만으론 탁수/벽클로즈업이 고평가되는 문제 보정.

    depth_conf는 '모델 확신'이지 '관벽 노출'이 아님(균일한 탁수 덩어리에 높게 나옴).
    관축 후보(카메라 광학축=검사 카메라는 보어를 향함 / PCA 1주축)별로 직교평면
    원피팅을 시도해 최저 잔차 채택. 링(관벽)=잔차 낮음, 탁수 덩어리/벽 평면=높음.
    pipe_score(0~100) = 100 × ring × √커버리지 × (0.3+0.7×inlier) × (0.5+0.5×conf정규화)
    """
    pts = wp.reshape(-1, 3)
    cf = conf.reshape(-1)
    sel = cf > np.percentile(cf, 50)
    pts = pts[sel]
    if len(pts) < 500:
        return {'cyl_res': None, 'ang_cov': 0.0, 'pipe_score': 0, 'axis': None}
    axes = []
    if cam_forward is not None:
        axes.append(('cam', cam_forward / np.linalg.norm(cam_forward)))
    try:
        q = pts - pts.mean(0)
        w, v = np.linalg.eigh(np.cov(q.T))
        axes.append(('pca', v[:, -1]))
    except np.linalg.LinAlgError:
        pass
    best = None
    for name, ax in axes:
        r = _ring_fit(pts, ax)
        if r is None:
            continue
        if best is None or r[0] < best[1][0]:
            best = (name, r)
    if best is None:
        return {'cyl_res': None, 'ang_cov': 0.0, 'pipe_score': 0, 'axis': None}
    name, (rel_res, ang_cov, inlier_frac, slab_ratio) = best
    cmean = float(conf.mean())
    cfac = min(max((cmean - 1.0) / 1.5, 0.0), 1.0)
    ring = max(0.0, 1.0 - rel_res / 0.4)
    # 하드 게이트: 미세 호(커버리지)·안개 원뿔(슬랩)·저주파 죽(텍스처) 차단
    cov_gate = min(max((ang_cov - 0.25) / 0.5, 0.0), 1.0)
    cyl_gate = min(max((slab_ratio - 0.55) / 0.30, 0.0), 1.0)
    tex_gate = 1.0 if lapvar is None else min(max(lapvar / 50.0, 0.0), 1.0)
    bore_gate = 1.0 if bore_frac is None else min(max(bore_frac / 0.08, 0.0), 1.0)
    score = int(round(100 * ring * cov_gate * cyl_gate * tex_gate * bore_gate
                      * (0.3 + 0.7 * inlier_frac) * (0.5 + 0.5 * cfac)))
    return {'cyl_res': round(rel_res, 3), 'ang_cov': round(ang_cov, 2),
            'slab_ratio': round(slab_ratio, 2), 'lapvar': None if lapvar is None else round(lapvar, 1), 'bore_frac': None if bore_frac is None else round(bore_frac, 3),
            'pipe_score': score, 'axis': name}




# ═══ 부분원호 원통 피팅 + metric 전개 (A4 본론) ═══
def _fit_cylinder_partial(wp_all, conf_all, paths, mid, cam_forward, diameter_mm,
                          cam_center=None, measure_defects=True,
                          K=None, extri_mid=None, det_conf=0.30,
                          det_conf_nodule=0.20, min_hits=2,
                          detect_on_unwrap=False, extri_all=None):
    wp, conf, frame_path = wp_all[mid], conf_all[mid], paths[mid]
    """VGGT 포인트클라우드에 원통 피팅(축 2DOF 최적화+트리밍) → 관경 앵커로 metric화.

    부분 원호(원주 일부만 노출)에서도 성립. 반환: 피팅 파라미터(mm)·품질지표·
    벽면 텍스처 전개도(θ×z, mm 스케일).
    """
    from scipy.optimize import minimize
    H, W = conf.shape
    pts_all = wp.reshape(-1, 3)
    cf = conf.reshape(-1)
    sel = cf > np.percentile(cf, 50)
    idx = np.where(sel)[0]
    if len(idx) < 2000:
        return None
    sub = idx[np.linspace(0, len(idx) - 1, min(20000, len(idx))).astype(int)]
    P = pts_all[sub]
    a0 = cam_forward / np.linalg.norm(cam_forward)

    def basis(ax):
        tmp = np.array([1., 0, 0]) if abs(ax[0]) < 0.9 else np.array([0, 1., 0])
        u = np.cross(ax, tmp); u /= np.linalg.norm(u)
        return u, np.cross(ax, u)

    def eval_axis(ax):
        u, v = basis(ax)
        x, y = P @ u, P @ v
        m = np.ones(len(x), bool)
        cx = cy = R = None
        for _ in range(3):
            A = np.stack([x[m], y[m], np.ones(int(m.sum()))], 1)
            b = -(x[m] ** 2 + y[m] ** 2)
            try:
                sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            cx, cy = -sol[0] / 2, -sol[1] / 2
            R2 = cx * cx + cy * cy - sol[2]
            if R2 <= 0:
                return None
            R = float(np.sqrt(R2))
            r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            m2 = np.abs(r - R) < 0.25 * R
            if m2.sum() < 500:
                break
            if (m2 == m).all():
                m = m2
                break
            m = m2
        rel = float(np.sqrt(np.mean((r[m] - R) ** 2)) / R)
        return {"rel": rel, "R": R, "cx": cx, "cy": cy, "u": u, "v": v,
                "inl_frac": float(m.mean())}

    def ax_of(th):
        u, v = basis(a0)
        ax = a0 + th[0] * u + th[1] * v
        return ax / np.linalg.norm(ax)

    res = minimize(lambda th: (eval_axis(ax_of(th)) or {"rel": 1e3})["rel"],
                   [0.0, 0.0], method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-4, maxiter=60))
    ax = ax_of(res.x)
    fit = eval_axis(ax)
    if fit is None:
        return None
    u, v, R = fit["u"], fit["v"], fit["R"]
    c0 = fit["cx"] * u + fit["cy"] * v          # 축 위의 한 점
    scale = (diameter_mm / 2.0) / R              # mm per VGGT unit

    # ── 전체 픽셀 metric 좌표 → 벽면 전개 ──
    q = pts_all - c0
    t_ax = q @ ax
    rv = q - np.outer(t_ax, ax)
    r_i = np.linalg.norm(rv, axis=1)
    theta = np.arctan2(rv @ v, rv @ u)
    rows = np.repeat(np.arange(H), W)
    osd_ok = (rows > H * 0.12) & (rows < H * 0.90)   # OSD 상하 마스킹
    wall = sel & osd_ok & (np.abs(r_i - R) < 0.25 * R)
    if wall.sum() < 1000:
        return None
    th_w, t_w = theta[wall], t_ax[wall]
    x_mm = th_w * (diameter_mm / 2.0)            # 원주 방향 mm
    y_mm = t_w * scale                           # 축 방향 mm
    gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (W, H)).reshape(-1)[wall].astype(np.float32)

    x0, x1 = np.percentile(x_mm, 0.5), np.percentile(x_mm, 99.5)
    y0, y1 = np.percentile(y_mm, 0.5), np.percentile(y_mm, 99.5)
    span_x, span_y = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
    ppm = min(2.0, 1500.0 / span_x, 1500.0 / span_y)   # px per mm (캔버스 상한)
    Wc, Hc = int(span_x * ppm) + 1, int(span_y * ppm) + 1
    acc = np.zeros((Hc, Wc), np.float32)
    cnt = np.zeros((Hc, Wc), np.float32)
    xi = np.clip(((x_mm - x0) * ppm).astype(int), 0, Wc - 1)
    yi = np.clip(((y_mm - y0) * ppm).astype(int), 0, Hc - 1)
    np.add.at(acc, (yi, xi), gray)
    np.add.at(cnt, (yi, xi), 1)
    img = np.zeros((Hc, Wc), np.uint8)
    nz = cnt > 0
    img[nz] = (acc[nz] / cnt[nz]).astype(np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    holes = ((img == 0) & (cv2.dilate((img > 0).astype(np.uint8),
             np.ones((9, 9), np.uint8)) > 0)).astype(np.uint8)
    if holes.any():
        img = cv2.inpaint(img, holes, 3, cv2.INPAINT_TELEA)
    unwrap = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ── 역매핑 밀집 컬러 텍스처(가능 시): 캔버스 픽셀→원통 3D→카메라 투영→원본 샘플 ──
    # 순방향 스플랫의 점 성김을 해소. 관측 영역(스플랫 점유 팽창) 밖은 지어내지 않고 검게.
    if K is not None and extri_mid is not None:
        try:
            gxm = x0 + (np.arange(Wc, dtype=np.float32) + 0.5) / ppm     # mm(원주)
            gym = y0 + (np.arange(Hc, dtype=np.float32) + 0.5) / ppm     # mm(축)
            thg = gxm / (diameter_mm / 2.0)
            zg = gym / scale
            cth, sth = np.cos(thg), np.sin(thg)
            Pw = (c0[None, None, :]
                  + (R * cth)[None, :, None] * u[None, None, :]
                  + (R * sth)[None, :, None] * v[None, None, :]
                  + zg[:, None, None] * ax[None, None, :])
            Rm, tm = extri_mid[:3, :3], extri_mid[:3, 3]
            Pc = Pw @ Rm.T + tm
            zc = Pc[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                uu = (K[0, 0] * Pc[..., 0] / zc + K[0, 2]).astype(np.float32)
                vv = (K[1, 1] * Pc[..., 1] / zc + K[1, 2]).astype(np.float32)
            valid = ((zc > 1e-6) & (uu >= 0) & (uu < W - 1)
                     & (vv >= H * 0.12) & (vv < H * 0.90))
            obs = cv2.dilate((cnt > 0).astype(np.uint8),
                             np.ones((11, 11), np.uint8)).astype(bool)
            valid &= obs
            frame_col = cv2.resize(cv2.imread(frame_path), (W, H))
            dense = cv2.remap(frame_col, uu, vv, cv2.INTER_LINEAR)
            unwrap = np.where(valid[..., None], dense, 0).astype(np.uint8)
        except Exception as e:
            print(f"역매핑 실패(스플랫 유지): {e}")

    # mm 눈금(100mm 격자)
    for gx in range(int(x0 // 100) * 100, int(x1) + 1, 100):
        px = int((gx - x0) * ppm)
        if 0 <= px < Wc:
            cv2.line(unwrap, (px, 0), (px, Hc - 1), (60, 60, 200), 1)
    for gy in range(int(y0 // 100) * 100, int(y1) + 1, 100):
        py = int((gy - y0) * ppm)
        if 0 <= py < Hc:
            cv2.line(unwrap, (0, py), (Wc - 1, py), (60, 60, 200), 1)

    # ── 결함 검출: 느슨한 검출(전 프레임)→원통좌표 융합→2-of-N→클래스임계→기하서명 ──
    defects = []
    viz_masks = []      # mid 프레임 원출력 (원본/클라우드 오버레이용)
    unwrap_extra = []   # (실험) 전개도 2차 검출 후보
    if measure_defects:
        try:
            _ym = _get_yolo()
            try:
                _ym.model.to("cuda")
            except Exception:
                pass
            LOOSE = 0.10                       # 1차: 느슨하게 다 잡음(미탐 방지)
            ACCEPT = {"corrosion_nodule": max(0.05, float(det_conf_nodule)),
                      "coating_peel": max(0.05, float(det_conf))}
            S = len(paths)
            hitmaps, confmaps = {}, {}
            drsum, drcnt = {}, {}
            jac_contribs = {}
            for k in range(S):
                fr = cv2.imread(paths[k])
                if fr is None:
                    continue
                oh, ow = fr.shape[:2]
                res_y = _ym.predict(fr, imgsz=960, conf=LOOSE,
                                    verbose=False, device=0)[0]
                if res_y.masks is None:
                    continue
                wpk = wp_all[k].reshape(-1, 3)
                qk = wpk - c0
                tk = qk @ ax
                rvk = qk - np.outer(tk, ax)
                rk = np.linalg.norm(rvk, axis=1)
                thk = np.arctan2(rvk @ v, rvk @ u)
                wallk = np.abs(rk - R) < 0.25 * R
                # 픽셀별 실표면적(자코비안): dA = t²/(fx·fy·n³·cosθi)
                #   비닝-점유법은 원거리/사면(픽셀밀도<빈밀도)서 과소 — GT 검증서 확인
                dA_k = None
                if K is not None and extri_all is not None:
                    if "_n3grid" not in locals():
                        uu2, vv2 = np.meshgrid(np.arange(W, dtype=np.float32),
                                               np.arange(H, dtype=np.float32))
                        ddx = (uu2 - K[0, 2]) / K[0, 0]
                        ddy = (vv2 - K[1, 2]) / K[1, 1]
                        _n3grid = (np.sqrt(ddx**2 + ddy**2 + 1.0) ** 3).reshape(-1)
                    cam_k = -extri_all[k][:3, :3].T @ extri_all[k][:3, 3]
                    ray = wpk - cam_k
                    tdist = np.linalg.norm(ray, axis=1).clip(1e-9, None)
                    radial = rvk / rk[:, None].clip(1e-9, None)
                    cosi_k = np.abs((ray / tdist[:, None] * -radial).sum(1)).clip(0.05, 1)
                    dA_k = (tdist ** 2) / (float(K[0, 0]) * float(K[1, 1])
                                           * _n3grid * cosi_k) * (scale ** 2)
                names = res_y.names
                frame_occ = {}   # cls -> 이 프레임의 합집합(프레임당 hit 1회)
                frame_pix = {}   # cls -> 자코비안용 픽셀 합집합(인스턴스 중복 방지)
                for i2 in range(len(res_y.boxes)):
                    cls = names[int(res_y.boxes.cls[i2])]
                    dconf = float(res_y.boxes.conf[i2])
                    poly = res_y.masks.xy[i2]
                    if poly is None or len(poly) < 3:
                        continue
                    mimg = np.zeros((H, W), np.uint8)
                    pl = np.asarray(poly, np.float32)
                    pl[:, 0] *= W / ow
                    pl[:, 1] *= H / oh
                    cv2.fillPoly(mimg, [pl.astype(np.int32)], 1)
                    if k == mid:
                        viz_masks.append((cls, mimg.astype(bool)))
                    sel_i = mimg.reshape(-1).astype(bool) & wallk
                    if sel_i.sum() < 20:
                        continue
                    dxk = np.clip(((thk[sel_i] * (diameter_mm / 2.0) - x0)
                                   * ppm).astype(int), 0, Wc - 1)
                    dyk = np.clip(((tk[sel_i] * scale - y0)
                                   * ppm).astype(int), 0, Hc - 1)
                    occ = np.zeros((Hc, Wc), np.uint8)
                    occ[dyk, dxk] = 1
                    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(
                                               cv2.MORPH_ELLIPSE, (7, 7)))
                    fo = frame_occ.setdefault(cls, np.zeros((Hc, Wc), np.uint8))
                    np.maximum(fo, occ, out=fo)
                    cmx = confmaps.setdefault(cls, np.zeros((Hc, Wc), np.float32))
                    np.maximum(cmx, occ.astype(np.float32) * dconf, out=cmx)
                    dsm = drsum.setdefault(cls, np.zeros((Hc, Wc), np.float32))
                    dcm = drcnt.setdefault(cls, np.zeros((Hc, Wc), np.float32))
                    np.add.at(dsm, (dyk, dxk), (rk[sel_i] - R) * scale)
                    np.add.at(dcm, (dyk, dxk), 1)
                    if dA_k is not None:
                        fp = frame_pix.setdefault(cls, np.zeros(H * W, bool))
                        fp |= sel_i   # 겹치는 인스턴스 중복 적산 방지(픽셀 합집합)
                for cls_f, fo in frame_occ.items():
                    hm = hitmaps.setdefault(cls_f, np.zeros((Hc, Wc), np.uint16))
                    hm += fo
                if dA_k is not None:
                    for cls_f, fp in frame_pix.items():
                        if not fp.any():
                            continue
                        dxk2 = np.clip(((thk[fp] * (diameter_mm / 2.0) - x0)
                                        * ppm).astype(int), 0, Wc - 1)
                        dyk2 = np.clip(((tk[fp] * scale - y0)
                                        * ppm).astype(int), 0, Hc - 1)
                        jac_contribs.setdefault(cls_f, []).append(
                            (k, dyk2, dxk2, dA_k[fp]))
            try:
                _ym.model.to("cpu")
                torch.cuda.empty_cache()
            except Exception:
                pass
            # 프레임 벽면 dr 중앙값 = 기준면 바이어스(부분원호+대편심서 R 과소추정 경향 보정)
            dr_med = float(np.median((r_i[wall] - R)) * scale) if wall.any() else 0.0
            # 영역화: min_hits 프레임 이상 지지(프레임 적으면 완화) + 클래스 임계
            eff_min = max(1, min(int(min_hits), S))
            for cls, hm in hitmaps.items():
                strong = (hm >= eff_min).astype(np.uint8)
                nlab, labs = cv2.connectedComponents(strong)
                for Lb in range(1, nlab):
                    regm = labs == Lb
                    if int(regm.sum()) < 25:
                        continue
                    maxc = float(confmaps[cls][regm].max())
                    if maxc < ACCEPT.get(cls, float(det_conf)):
                        continue
                    area_bin = float(regm.sum()) / (ppm * ppm)
                    area_mm2 = area_bin
                    if jac_contribs.get(cls):
                        per_frame = {}
                        for (kf, dyk2, dxk2, dAv) in jac_contribs[cls]:
                            inm = regm[dyk2, dxk2]
                            if inm.any():
                                per_frame[kf] = per_frame.get(kf, 0.0) + float(dAv[inm].sum())
                        if per_frame:
                            area_mm2 = max(per_frame.values())   # 가장 완전히 본 프레임
                    seen = int(hm[regm].max())
                    dcsum = float(drcnt[cls][regm].sum())
                    dr_mean = float(drsum[cls][regm].sum() / max(dcsum, 1.0)) - dr_med
                    # 기하 서명(프레임 기준면 대비 상대): 결절=돌출(dr<0) 기대 — 강한 모순만 경고
                    geom_warn = bool(cls == "corrosion_nodule" and dr_mean > 5.0)
                    col = DEFECT_COLORS.get(cls, (0, 255, 255))
                    cts, _ = cv2.findContours(regm.astype(np.uint8),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)
                    ov = unwrap.copy()
                    cv2.drawContours(ov, cts, -1, col, -1)
                    unwrap[:] = cv2.addWeighted(ov, 0.30, unwrap, 0.70, 0)
                    cv2.drawContours(unwrap, cts, -1, col, 2)
                    ys2, xs2 = np.where(regm)
                    cv2.putText(unwrap, "%s %.0fcm2 x%d" % (
                        DEFECT_LABEL_EN.get(cls, cls), area_mm2 / 100.0, seen),
                        (max(int(xs2.min()), 2), max(int(ys2.min()) - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
                    defects.append({"cls": DEFECT_NAMES_KO.get(cls, cls),
                                    "conf": round(maxc, 2),
                                    "area_mm2": round(area_mm2, 0),
                                    "area_bin_mm2": round(area_bin, 0),
                                    "frames_seen": seen, "n_frames": S,
                                    "dr_mm": round(dr_mean, 1),
                                    "geom_warn": geom_warn})
            # (실험) 전개도 공간 2차 검출 — 경사 미탐 탐색, 보고만
            if detect_on_unwrap:
                try:
                    _ym.model.to("cuda")
                    ru = _ym.predict(unwrap, imgsz=960, conf=LOOSE,
                                     verbose=False, device=0)[0]
                    _ym.model.to("cpu")
                    torch.cuda.empty_cache()
                    if ru.masks is not None:
                        for i3 in range(len(ru.boxes)):
                            cls = ru.names[int(ru.boxes.cls[i3])]
                            unwrap_extra.append({
                                "cls": DEFECT_NAMES_KO.get(cls, cls),
                                "conf": round(float(ru.boxes.conf[i3]), 2),
                                "area_mm2_est": round(float(cv2.contourArea(
                                    np.asarray(ru.masks.xy[i3], np.float32)))
                                    / (ppm * ppm), 0)
                                if ru.masks.xy[i3] is not None
                                and len(ru.masks.xy[i3]) >= 3 else None})
                except Exception as e:
                    unwrap_extra = [{"error": str(e)}]
        except Exception as e:
            defects = [{"error": f"결함 투영 실패: {e}"}]

    out = {
        "dr_bias_mm": round(dr_med, 1) if measure_defects else None,
        "unwrap_extra": unwrap_extra,
        "ppm": round(ppm, 3),                    # px per mm (파노라마 배치용)
        "circ_x0_mm": round(float(x0), 1),       # 원주 시작(θ·R mm, 공통 θ기준)
        "defects": defects,
        "defect_total_mm2": {c: round(sum(d["area_mm2"] for d in defects
                                          if d.get("area_mm2") and d.get("cls") == c), 0)
                             for c in set(d.get("cls") for d in defects
                                          if d.get("area_mm2"))},
        "scale_mm_per_unit": round(scale, 2),
        "radius_est_units": round(R, 5),
        "residual_mm_rms": round(fit["rel"] * (diameter_mm / 2.0), 1),
        "residual_rel": round(fit["rel"], 3),
        "inlier_frac": round(fit["inl_frac"], 2),
        "arc_coverage_deg": round(float(np.ptp(th_w)) * 180 / np.pi, 1),
        "axial_extent_mm": round(float(span_y), 0),
        "circ_extent_mm": round(float(span_x), 0),
        "axis_tilt_from_cam_deg": round(float(np.degrees(
            np.arccos(np.clip(abs(ax @ a0), -1, 1)))), 1),
        "unwrap_b64": b64jpg(unwrap, 88),
    }
    if cam_center is not None:
        d = cam_center - c0
        ecc = np.linalg.norm(d - (d @ ax) * ax) * scale
        out["cam_ecc_mm"] = round(float(ecc), 1)
        out["cam_ecc_ratio"] = round(float(ecc / (diameter_mm / 2.0)), 2)
    return out, viz_masks, wall




class PanoReq(BaseModel):
    video_path: str
    diameter_mm: int
    frames: list = []       # 명시 스톱 프레임(비면 균등 n_stops)
    n_stops: int = 8
    min_score: int = 15     # 사전 관다움 게이트(낮춤) — 최종 판정은 피팅 품질 게이트
    det_conf: float = 0.30  # 박리 채택 임계
    det_conf_nodule: float = 0.20
    min_hits: int = 2
    n_frames: int = 4
    step: int = 12
    start_frame: int = 0
    end_frame: int = 0


@app.post('/panorama')
def panorama(req: PanoReq):
    try:
        return _pano_impl(req)
    finally:
        release_model()


def _pano_impl(req: PanoReq):
    """여러 스톱을 원통 피팅·역매핑 전개 후 이어붙여 전관 컬러 전개 시트 생성.

    원주(θ)는 공통 기준으로 절대 배치. 축 방향은 오도메트리가 없어
    프레임 순서 스택(스트립별 축범위 mm) — 연속 지도가 아닌 구간별 metric 시트.
    스톱 품질 게이트: 관다움 S>=25 + depth 반전 아님.
    """
    import tempfile
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.load_fn import load_and_preprocess_images
    p = os.path.realpath(req.video_path)
    if not any(p.startswith(os.path.realpath(r) + os.sep) for r in ALLOWED_ROOTS):
        return {'success': False, 'error': '허용 경로 밖 파일'}
    if not os.path.isfile(p):
        return {'success': False, 'error': f'파일 없음: {p}'}
    D = int(req.diameter_mm)
    if D <= 0:
        return {'success': False, 'error': '관경(diameter_mm) 필요'}

    cap = cv2.VideoCapture(p)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if req.frames:
        centers = [int(c) for c in req.frames]
    else:
        s0 = max(0, int(req.start_frame))
        e0 = int(req.end_frame) if req.end_frame and int(req.end_frame) > s0 else total - 1
        n = max(2, min(int(req.n_stops), 20))
        centers = list(np.linspace(s0 + (e0 - s0) * 0.05,
                                   s0 + (e0 - s0) * 0.95, n).astype(int))
    nfr = max(2, min(int(req.n_frames), 6))
    step = max(1, int(req.step))
    model = get_model()
    torch.cuda.empty_cache()

    strips, skipped = [], []
    totals = {}
    for c in centers:
        tmpd = tempfile.mkdtemp(prefix='recon3d_pano_')
        paths = []
        half = nfr // 2
        for i, fi in enumerate(range(c - half * step, c - half * step + nfr * step, step)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(fi, total - 1)))
            ret, fr = cap.read()
            if not ret:
                continue
            fp = f"{tmpd}/{i:02d}.jpg"
            cv2.imwrite(fp, fr)
            paths.append(fp)
        if len(paths) < 2:
            skipped.append({'frame': int(c), 'reason': '프레임 추출 실패'})
            continue
        try:
            images = load_and_preprocess_images(paths).to("cuda")
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(images)
            wp_np = pred['world_points'].squeeze(0).float().cpu().numpy()
            cf_np = pred['depth_conf'].squeeze(0).float().cpu().numpy()
            d_np = pred['depth'].squeeze(0).float().cpu().numpy()
            ex, kin = pose_encoding_to_extri_intri(pred['pose_enc'], images.shape[-2:])
            ex = ex.squeeze(0).float().cpu().numpy()
            kin = kin.squeeze(0).float().cpu().numpy()
            del pred, images
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            skipped.append({'frame': int(c), 'reason': 'OOM'})
            continue
        mid = wp_np.shape[0] // 2
        fwd = ex[mid][:3, :3].T @ np.array([0., 0., 1.])
        lap, bore = _frame_stats(paths[mid])
        met = _pipe_metrics(wp_np[mid], cf_np[mid], cam_forward=fwd,
                            lapvar=lap, bore_frac=bore)
        pc = _brightness_corr(d_np[mid, :, :, 0], paths[mid])
        if pc is not None and pc > INV_CORR_THR:
            skipped.append({'frame': int(c), 'reason': f'depth 반전 의심({pc:+.2f})'})
            continue
        if met['pipe_score'] < max(5, int(req.min_score)):
            skipped.append({'frame': int(c), 'reason': f"관다움 미달(S{met['pipe_score']})"})
            continue
        cam_c = -ex[mid][:3, :3].T @ ex[mid][:3, 3]
        try:
            ret2 = _fit_cylinder_partial(wp_np, cf_np, paths, mid,
                                         fwd, D, cam_center=cam_c,
                                         measure_defects=True,
                                         K=kin[mid], extri_mid=ex[mid],
                                         extri_all=ex,
                                         det_conf=req.det_conf,
                                         det_conf_nodule=req.det_conf_nodule,
                                         min_hits=req.min_hits)
        except Exception as e:
            skipped.append({'frame': int(c), 'reason': f'피팅 예외: {e}'})
            continue
        if not ret2:
            skipped.append({'frame': int(c), 'reason': '피팅 실패(포인트 부족)'})
            continue
        out, _, _ = ret2
        # 최종 품질 게이트: 관다움을 낮춘 대신 피팅 잔차·인라이어로 판정
        if out['residual_rel'] > 0.22 or out['inlier_frac'] < 0.55:
            skipped.append({'frame': int(c),
                            'reason': f"피팅 품질 미달(res{out['residual_rel']}·inl{out['inlier_frac']})"})
            continue
        img = cv2.imdecode(np.frombuffer(base64.b64decode(out['unwrap_b64']),
                                         np.uint8), cv2.IMREAD_COLOR)
        for dd in out.get('defects', []):
            if dd.get('area_mm2'):
                totals[dd['cls']] = totals.get(dd['cls'], 0) + dd['area_mm2']
        strips.append({'frame': int(c), 'score': met['pipe_score'],
                       'img': img, 'ppm': out['ppm'], 'x0': out['circ_x0_mm'],
                       'axial_mm': out['axial_extent_mm'],
                       'residual_mm': out['residual_mm_rms'],
                       'defects': out.get('defects', [])})
        if len(strips) % 3 == 0:
            torch.cuda.empty_cache()
    cap.release()

    if not strips:
        return {'success': False, 'error': '성립 스톱 없음', 'skipped': skipped}

    # ── 공통 원주축 조립 (θ·R ∈ [-πR, πR]) ──
    PPM = 0.8
    R_mm = D / 2.0
    Wp = int(np.pi * D * PPM) + 2
    HEAD = 26
    rows = []
    for st in strips:
        f = PPM / st['ppm']
        im = cv2.resize(st['img'], (max(1, int(st['img'].shape[1] * f)),
                                    max(1, int(st['img'].shape[0] * f))))
        row = np.zeros((im.shape[0] + HEAD, Wp, 3), np.uint8)
        offx = int((st['x0'] + np.pi * R_mm) * PPM)
        offx = max(0, min(offx, Wp - im.shape[1])) if im.shape[1] < Wp else 0
        row[HEAD:, offx:offx + min(im.shape[1], Wp)] = im[:, :min(im.shape[1], Wp)]
        _ko2en = {"결절": "NODULE", "박리": "PEEL"}   # cv2 putText는 ASCII만
        dtxt = " ".join("%s %.0fcm2" % (_ko2en.get(k, "DEFECT"), v / 100.0)
                        for k, v in
                        {d['cls']: d['area_mm2'] for d in st['defects']
                         if d.get('area_mm2')}.items()) or "-"
        cv2.putText(row, "f%d  S%d  axial %dmm  res %.0fmm  %s" %
                    (st['frame'], st['score'], st['axial_mm'],
                     st['residual_mm'], dtxt),
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 220, 255), 1)
        cv2.line(row, (0, HEAD - 2), (Wp, HEAD - 2), (70, 70, 70), 1)
        rows.append(row)
    pano = np.vstack(rows)
    # 상단 원주 눈금(0.5m 간격 mm)
    return {'success': True, 'pano_b64': b64jpg(pano, 87),
            'n_strips': len(strips), 'skipped': skipped,
            'defect_total_mm2': {k: round(v, 0) for k, v in totals.items()},
            'note': '축방향은 프레임 순서 스택(스톱 간 절대거리 미상) — 스트립 내부만 metric. '
                    '결함 합계는 스톱 간 중복 가능성 있음(간격이 좁을 때).',
            'strips': [{k: v for k, v in st.items() if k != 'img'} for st in strips]}


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
    try:
        return _scan_impl(req)
    finally:
        release_model()   # 유휴 VRAM 반납


def _scan_impl(req: ScanReq):
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
    torch.cuda.empty_cache()   # 동거 프로세스(5004/8085)로 VRAM 빠듯 — 스캔 전 회수
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
        def _infer_point():
            images = load_and_preprocess_images(paths).to("cuda")
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                pred = model(images)
            cmean = float(pred['depth_conf'].float().mean())
            wp_np = pred['world_points'].squeeze(0).float().cpu().numpy()
            cf_np = pred['depth_conf'].squeeze(0).float().cpu().numpy()
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri
            _ex, _ = pose_encoding_to_extri_intri(pred['pose_enc'], images.shape[-2:])
            _ex = _ex.squeeze(0).float().cpu().numpy()
            _mid = wp_np.shape[0] // 2
            _fwd = _ex[_mid][:3, :3].T @ np.array([0., 0., 1.])
            _lap2, _bore2 = _frame_stats(paths[_mid])
            met = _pipe_metrics(wp_np[_mid], cf_np[_mid], cam_forward=_fwd,
                                lapvar=_lap2, bore_frac=_bore2)
            _dmid = pred['depth'].squeeze(0).float().cpu().numpy()[_mid, :, :, 0]
            _pc = _brightness_corr(_dmid, paths[_mid])
            if _pc is not None and _pc > INV_CORR_THR:
                met['pipe_score'] = min(met['pipe_score'], 10)
                met['inv'] = True
            del pred, images
            return cmean, met
        try:
            try:
                cmean, met = _infer_point()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()          # VRAM 빠듯(동거 프로세스) — 1회 재시도
                cmean, met = _infer_point()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"scan f{c}: OOM (재시도 실패)")
            points.append({'frame': int(c), 'conf': None, 'score': None})
            continue
        except Exception as e:
            print(f"scan f{c}: {type(e).__name__}: {e}")
            points.append({'frame': int(c), 'conf': None, 'score': None})
            continue
        if ci % 8 == 7:
            torch.cuda.empty_cache()
        points.append({'frame': int(c), 'conf': round(cmean, 2),
                       'score': met['pipe_score'], 'cyl_res': met['cyl_res'],
                       'ang_cov': met['ang_cov'], 'inv': bool(met.get('inv', False))})
    cap.release()
    torch.cuda.empty_cache()

    valid = [pt for pt in points if pt.get('score') is not None]
    best = sorted(valid, key=lambda x: -x['score'])[:5]
    return {'success': True, 'total_frames': total, 'points': points,
            'best': best, 'scan_sec': round(time.time() - t0, 1),
            'n_frames': nfr, 'step': step}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5006)
