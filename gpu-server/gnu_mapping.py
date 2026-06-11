#!/usr/bin/env python3
"""
GNU HydrosystemLAB 매핑 모듈 (독립)

PPNet (카메라 자세 추정) + 3D→2D 투영 매핑 알고리즘
원본: https://github.com/GNU-HydrosystemLAB/Performance_indicator

이 모듈은 기존 PipeUnwrapper와 독립적으로 동작하며,
두 방식의 결과를 비교할 수 있도록 설계되었다.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import os
import base64

# ─── 카메라 파라미터 ───
CAMERA_PARAMS = {
    "HD":  {"f": 934.89,  "cx": 640, "cy": 360, "size": (1280, 720)},
    "FHD": {"f": 1479.749, "cx": 960, "cy": 540, "size": (1920, 1080)},
}

# ─── 관종별 직경 프리셋 (mm) ───
# group = "field"  : 실제 현장 관경 (K-water 영상 기반 / run_train.py)
# group = "paper"  : Performance_indicator 논문 평가용 실험 시편
PIPE_DIMENSIONS = {
    # ── 현장 관경 ──
    "DCIP_80":  {"diameter_mm": 80,   "water_default": False, "group": "field", "label": "DCIP 80mm"},
    "DCIP_100": {"diameter_mm": 100,  "water_default": False, "group": "field", "label": "DCIP 100mm"},
    "DCIP_150": {"diameter_mm": 150,  "water_default": False, "group": "field", "label": "DCIP 150mm"},
    "DCIP_200": {"diameter_mm": 200,  "water_default": False, "group": "field", "label": "DCIP 200mm"},
    "DCIP_300": {"diameter_mm": 300,  "water_default": False, "group": "field", "label": "DCIP 300mm"},
    "DCIP_500": {"diameter_mm": 500,  "water_default": False, "group": "field", "label": "DCIP 500mm"},
    "DCIP_700": {"diameter_mm": 700,  "water_default": False, "group": "field", "label": "DCIP 700mm"},
    "CIP_100":  {"diameter_mm": 100,  "water_default": False, "group": "field", "label": "CIP 100mm"},
    "PE_200":   {"diameter_mm": 200,  "water_default": False, "group": "field", "label": "PE 200mm"},
    "SP_700":   {"diameter_mm": 700,  "water_default": False, "group": "field", "label": "SP 700mm"},

    # ── 논문 시편 (Performance_indicator 재현용) ──
    "CIP": {"diameter_mm": 23,   "water_default": False, "group": "paper", "label": "CIP 23mm (논문 시편)"},
    "PVC": {"diameter_mm": 83,   "water_default": True,  "group": "paper", "label": "PVC 83mm (논문, 만관)"},
    "CP":  {"diameter_mm": 84.6, "water_default": False, "group": "paper", "label": "CP 84.6mm (논문 시편)"},
    "PP":  {"diameter_mm": 53,   "water_default": False, "group": "paper", "label": "PP 53mm (논문 시편)"},
}

PIPE_GROUP_LABELS = {
    "field": "현장 관경",
    "paper": "논문 시편 (GNU 평가용)",
}


def detect_resolution(width, height):
    """이미지 해상도에서 카메라 파라미터 키 반환"""
    for key, params in CAMERA_PARAMS.items():
        if params["size"] == (width, height):
            return key
    # 가장 가까운 해상도 선택
    return "FHD" if width > 1280 else "HD"


# ═══════════════════════════════════════════
#  PPNet — 카메라 자세 추정
# ═══════════════════════════════════════════
class PPNet:
    """Pipe Pose Network — ResNet-101 기반 카메라 자세 회귀 모델

    출력: [vp_x_px, vp_y_px, angle_deg, step]
    """

    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def run(self, frame_rgb):
        """카메라 자세 추정

        Args:
            frame_rgb: RGB numpy array (H, W, 3)

        Returns:
            {
              'pose': [vp_x, vp_y, angle, step],
              'raw': [x%, y%, tx, ty],
              'shape': (H, W, C),
            }
        """
        frame_pil = Image.fromarray(frame_rgb)
        input_tensor = self.transform(frame_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = output.squeeze(0).tolist()

        pose = self._convert_output(frame_rgb.shape, prediction)
        return {
            'pose': pose,
            'raw': prediction,
            'shape': frame_rgb.shape,
        }

    def _convert_output(self, shape, prediction):
        """모델 출력 [x%, y%, tx, ty] → [vp_x_px, vp_y_px, angle_deg, step]"""
        x_pct, y_pct, tx, ty = prediction

        s = np.sqrt(tx**2 + ty**2)
        if s < 1e-6:
            angle = 0.0
        else:
            d = np.rad2deg(np.arccos(np.clip(tx / s, -1, 1)))
            if d < 0:
                d += 180
            elif d > 180:
                d -= 180
            if ty < 0:
                d = -d
            angle = d

        vp_x = x_pct * shape[1] / 100
        vp_y = y_pct * shape[0] / 100
        step = max(s / 100, 0.01)

        return [vp_x, vp_y, angle, step]


# ═══════════════════════════════════════════
#  PipeMapper — 3D→2D 투영 매핑
# ═══════════════════════════════════════════
class PipeMapper:
    """3D 원통 → 2D 투영 기반 관 내부 전개도 생성

    핵심 알고리즘:
      1. 3D 원통 메시 생성 (반지름 R, 길이 L)
      2. 카메라 내부/외부 파라미터로 3D→2D 투영
      3. 원본 이미지에서 색상 샘플링 → 전개도 생성
    """

    def __init__(self, f=934.89, cx=640, cy=360,
                 pipe_diameter_mm=80, water=False,
                 pixel_per_mm=10.0, max_depth_mm=300,
                 device=None):
        """
        Args:
            f: 카메라 초점거리 (px)
            cx, cy: 카메라 주점
            pipe_diameter_mm: 관경 (mm)
            water: 수중 촬영 여부 (True이면 f *= 1.33)
            pixel_per_mm: 전개도 해상도 (px/mm)
            max_depth_mm: 전개도 최대 깊이 (mm)
        """
        self.f = f * 1.33 if water else f
        self.cx = cx
        self.cy = cy
        self.diameter = pipe_diameter_mm
        self.radius = pipe_diameter_mm / 2
        self.water = water
        self.pixel_per_mm = pixel_per_mm
        self.max_depth_mm = max_depth_mm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 전개도 크기
        self.out_h = int(max_depth_mm * pixel_per_mm)      # 세로: 깊이
        self.out_w = int(pipe_diameter_mm * np.pi * pixel_per_mm)  # 가로: 둘레

        # 3D 원통 메시 생성
        self._build_3d_mesh()

    def _build_3d_mesh(self):
        """3D 원통 표면 메시 생성"""
        z_values = np.linspace(0, self.max_depth_mm, self.out_h)
        angles = np.linspace(0, 2 * np.pi, self.out_w)

        x_coords = self.radius * np.cos(angles)
        y_coords = self.radius * np.sin(angles)

        x_grid, z_grid = np.meshgrid(x_coords, z_values)
        y_grid, _ = np.meshgrid(y_coords, z_values)

        points = np.stack([
            x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), np.ones(x_grid.size)
        ], axis=1)

        self.points_3d = torch.tensor(points, dtype=torch.float32, device=self.device)

    def _project_to_2d(self, alpha, beta, gamma, tx, ty, tz):
        """3D 점들을 2D 이미지 좌표로 투영"""
        f = self.f

        K = torch.tensor([
            [f, 0, self.cx],
            [0, f, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)

        # 회전 행렬 Rz @ Ry @ Rx
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        cb, sb = torch.cos(beta), torch.sin(beta)
        cg, sg = torch.cos(gamma), torch.sin(gamma)

        R_x = torch.tensor([[1, 0, 0], [0, ca, -sa], [0, sa, ca]],
                            dtype=torch.float32, device=self.device)
        R_y = torch.tensor([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]],
                            dtype=torch.float32, device=self.device)
        R_z = torch.tensor([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]],
                            dtype=torch.float32, device=self.device)

        R = R_z @ R_y @ R_x
        T = torch.tensor([tx, ty, tz], dtype=torch.float32, device=self.device)

        points_cam = torch.matmul(self.points_3d[:, :3], R.T) + T
        points_proj = torch.matmul(points_cam, K.T)
        points_2d = points_proj[:, :2] / points_proj[:, 2:3].clamp(min=1e-6)

        return points_2d.to(torch.int32)

    def unwrap(self, img, pose):
        """이미지를 전개도로 변환

        Args:
            img: BGR 또는 RGB numpy array (H, W, 3) 또는 grayscale (H, W)
            pose: [vp_x, vp_y, angle_deg, step] — PPNet 출력

        Returns:
            전개도 numpy array (out_h, out_w, 3) 또는 (out_h, out_w)
        """
        vp_x, vp_y, degrees, step = pose

        # VP → 회전각
        vp_x_t = torch.tensor(vp_x, dtype=torch.float32, device=self.device)
        vp_y_t = torch.tensor(vp_y, dtype=torch.float32, device=self.device)
        cx_t = torch.tensor(self.cx, dtype=torch.float32, device=self.device)
        cy_t = torch.tensor(self.cy, dtype=torch.float32, device=self.device)
        f_t = torch.tensor(self.f, dtype=torch.float32, device=self.device)

        alpha = torch.atan((cy_t - vp_y_t) / f_t)
        beta = torch.atan((vp_x_t - cx_t) / f_t)
        gamma = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        # 카메라 이동
        step_t = torch.tensor(step, dtype=torch.float32, device=self.device)
        radius_t = torch.tensor(self.radius, dtype=torch.float32, device=self.device)
        radians = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32, device=self.device))
        tx = -(step_t * radius_t * torch.cos(radians))
        ty = -(step_t * radius_t * torch.sin(radians))
        tz = torch.tensor(-1.0, dtype=torch.float32, device=self.device)

        # 투영
        points_2d = self._project_to_2d(alpha, beta, gamma, tx, ty, tz)

        # 이미지에서 샘플링
        img_tensor = torch.tensor(img, dtype=torch.uint8, device=self.device)

        if len(img_tensor.shape) == 3:
            height, width, _ = img_tensor.shape
            valid = ((0 <= points_2d[:, 0]) & (points_2d[:, 0] < width) &
                     (0 <= points_2d[:, 1]) & (points_2d[:, 1] < height))

            result = torch.zeros((points_2d.shape[0], 3), dtype=torch.uint8, device=self.device)
            result[valid] = img_tensor[points_2d[valid, 1], points_2d[valid, 0]]
            result = result.view(self.out_h, self.out_w, 3)
        else:
            height, width = img_tensor.shape
            valid = ((0 <= points_2d[:, 0]) & (points_2d[:, 0] < width) &
                     (0 <= points_2d[:, 1]) & (points_2d[:, 1] < height))

            result = torch.zeros((points_2d.shape[0],), dtype=torch.uint8, device=self.device)
            result[valid] = img_tensor[points_2d[valid, 1], points_2d[valid, 0]]
            result = result.view(self.out_h, self.out_w)

        return result.cpu().numpy()

    def compute_depth_map(self, pose, img_shape, return_cos_alpha=False):
        """각 이미지 픽셀에서 카메라 → 관벽까지의 거리(mm) 맵 계산

        수식:
          - 카메라 ray in cam frame: d_cam = K⁻¹ · [u, v, 1]
          - world ray: P(λ) = cam_origin + λ · R⁻¹·d_cam
          - 원통 x² + y² = r² 와 교차 → 이차방정식 해
          - cos α = |d_world · wall_normal| / ||d_world||
            (wall_normal = 원통 표면 법선, 반경 방향)

        Returns:
            depth_mm: (H, W) float32 — 거리(mm), 관벽 밖은 NaN
            (return_cos_alpha=True일 때) (depth_mm, cos_alpha): 각 픽셀의 |cos α|
        """
        vp_x, vp_y, degrees, step = pose
        h = int(img_shape[0])
        w = int(img_shape[1])

        f = float(self.f); cx = float(self.cx); cy = float(self.cy)
        r = float(self.radius); max_L = float(self.max_depth_mm)

        # Rotation
        alpha = np.arctan((cy - vp_y) / f)
        beta = np.arctan((vp_x - cx) / f)
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
        Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float64)
        R_mat = Ry @ Rx  # gamma=0 → Rz=I

        rad = np.deg2rad(float(degrees))
        tx = -float(step) * r * np.cos(rad)
        ty = -float(step) * r * np.sin(rad)
        tz = -1.0
        T = np.array([tx, ty, tz], dtype=np.float64)

        R_inv = R_mat.T
        cam_origin = -R_inv @ T            # (3,)

        # pixel grid → camera-frame rays
        u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float64),
                                     np.arange(h, dtype=np.float64))
        du = (u_grid - cx) / f
        dv = (v_grid - cy) / f
        d_cam = np.stack([du, dv, np.ones_like(du)], axis=-1)  # (H,W,3)

        # world ray direction: d_world = R⁻¹ · d_cam
        d_world = np.einsum('ij,hwj->hwi', R_inv, d_cam)       # (H,W,3)
        dx, dy, dz = d_world[..., 0], d_world[..., 1], d_world[..., 2]

        wx, wy, wz = cam_origin
        A = dx * dx + dy * dy
        B = 2.0 * (wx * dx + wy * dy)
        C = wx * wx + wy * wy - r * r
        disc = B * B - 4.0 * A * C

        sqrt_disc = np.sqrt(np.maximum(disc, 0))
        with np.errstate(invalid='ignore', divide='ignore'):
            lam_p = (-B + sqrt_disc) / (2.0 * A)
            lam_n = (-B - sqrt_disc) / (2.0 * A)

        # 카메라는 원통 안에 있으므로 한 근은 양, 한 근은 음. 양의 근 채택.
        lam = np.where(lam_p > 0, lam_p, lam_n)
        valid = (disc >= 0) & (A > 1e-12) & (lam > 0) & np.isfinite(lam)

        iz = wz + lam * dz
        valid &= (iz >= 0.0) & (iz <= max_L)

        d_world_norm = np.sqrt(dx * dx + dy * dy + dz * dz)
        distance = lam * d_world_norm
        distance = np.where(valid, distance, np.nan).astype(np.float32)

        if not return_cos_alpha:
            return distance

        # cos α = |d_world · wall_normal| / ||d_world||
        # wall_normal at hit point P = -(P_x, P_y, 0) / R (inward)
        # |d_world · (−(P_x, P_y, 0)/R)| = |dx·P_x + dy·P_y| / R
        # where P_x = wx + λ·dx, P_y = wy + λ·dy
        # → |dx·(wx + λ·dx) + dy·(wy + λ·dy)| / R
        # = |(dx·wx + dy·wy) + λ·(dx² + dy²)| / R
        # = |B/2 + λ·A| / R     (B, A 재사용)
        numerator = np.abs(B / 2.0 + lam * A)
        with np.errstate(invalid='ignore', divide='ignore'):
            cos_alpha = numerator / (r * np.maximum(d_world_norm, 1e-9))
        cos_alpha = np.where(valid, cos_alpha, np.nan).astype(np.float32)
        return distance, cos_alpha

    @staticmethod
    def render_depth_heatmap(depth_mm, vmin=None, vmax=None):
        """depth map → BGR 컬러맵. NaN은 검정."""
        valid = np.isfinite(depth_mm)
        if not np.any(valid):
            return np.zeros((*depth_mm.shape, 3), dtype=np.uint8), (None, None)
        d_valid = depth_mm[valid]
        lo = float(vmin) if vmin is not None else float(np.nanpercentile(d_valid, 2))
        hi = float(vmax) if vmax is not None else float(np.nanpercentile(d_valid, 98))
        if hi - lo < 1e-6:
            hi = lo + 1.0
        norm = np.clip((depth_mm - lo) / (hi - lo), 0, 1)
        norm = np.where(valid, norm, 0)
        heat = (norm * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        heat_color[~valid] = 0
        return heat_color, (lo, hi)

    def get_coordinate_system(self):
        """좌표계 메타데이터"""
        circumference = np.pi * self.diameter
        return {
            'method': 'GNU 3D Projection',
            'x_axis': 'circumferential',
            'y_axis': 'axial depth',
            'x_range_mm': round(circumference, 2),
            'y_range_mm': self.max_depth_mm,
            'mm_per_px_x': round(1.0 / self.pixel_per_mm, 4),
            'mm_per_px_y': round(1.0 / self.pixel_per_mm, 4),
            'output_width': self.out_w,
            'output_height': self.out_h,
            'pipe_diameter_mm': self.diameter,
            'focal_length': round(self.f, 2),
            'water': self.water,
        }


# ═══════════════════════════════════════════
#  통합 인터페이스
# ═══════════════════════════════════════════
class GNUMappingEngine:
    """PPNet + PipeMapper 통합 엔진

    사용법:
        engine = GNUMappingEngine(ppnet_model_path, pipe_diameter_mm=80)
        result = engine.process_frame(frame_rgb)
    """

    _ppnet_instance = None
    _ppnet_path = None

    def __init__(self, ppnet_model_path=None, pipe_diameter_mm=80,
                 water=False, pixel_per_mm=10.0, max_depth_mm=300):

        # PPNet 싱글톤 (모델 로딩 1회만)
        if ppnet_model_path and (GNUMappingEngine._ppnet_path != ppnet_model_path):
            GNUMappingEngine._ppnet_instance = PPNet(ppnet_model_path)
            GNUMappingEngine._ppnet_path = ppnet_model_path

        self.ppnet = GNUMappingEngine._ppnet_instance
        self.pipe_diameter_mm = pipe_diameter_mm
        self.water = water
        self.pixel_per_mm = pixel_per_mm
        self.max_depth_mm = max_depth_mm

    def process_frame(self, frame_bgr, defect_masks=None, include_depth_map=False,
                      pose_override=None, camera_override=None):
        """프레임 분석 — PPNet 추론 + 전개도 생성

        Args:
            frame_bgr: BGR numpy array
            defect_masks: [{label, mask(H,W)}] 리스트 (옵션)
            pose_override: (vp_x, vp_y, angle, step) 튜플. None 이면 PPNet 추론, 아니면 그 값 사용.

        Returns:
            {
              'pose': {vp_x, vp_y, angle, step, raw},
              'unwrapped_rgb': base64 jpg,
              'unwrapped_defects': [{label, unwrapped_mask_b64, area_px, area_mm2}],
              'coordinate_system': {...},
            }
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]

        # 카메라 파라미터 자동 결정
        res_key = detect_resolution(w, h)
        cam = dict(CAMERA_PARAMS[res_key])
        # 사용자 override (캘리브레이션용)
        if camera_override:
            if 'f' in camera_override and camera_override['f']:
                cam['f'] = float(camera_override['f'])
            if 'cx' in camera_override and camera_override['cx'] is not None:
                cam['cx'] = float(camera_override['cx'])
            if 'cy' in camera_override and camera_override['cy'] is not None:
                cam['cy'] = float(camera_override['cy'])

        # 자세: 사용자 override 가 있으면 PPNet 스킵
        if pose_override is not None:
            vp_x, vp_y, angle, step = (float(x) for x in pose_override)
            pose = [vp_x, vp_y, angle, step]
            ppnet_result = {'pose': pose, 'raw': None, 'pose_source': 'manual'}
        else:
            ppnet_result = self.ppnet.run(frame_rgb)
            pose = ppnet_result['pose']
            vp_x, vp_y, angle, step = pose
            ppnet_result['pose_source'] = 'ppnet'

        # PipeMapper 생성
        mapper = PipeMapper(
            f=cam['f'], cx=cam['cx'], cy=cam['cy'],
            pipe_diameter_mm=self.pipe_diameter_mm,
            water=self.water,
            pixel_per_mm=self.pixel_per_mm,
            max_depth_mm=self.max_depth_mm,
        )

        # RGB 전개도
        unwrapped = mapper.unwrap(frame_rgb, pose)
        unwrapped_bgr = cv2.cvtColor(unwrapped, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', unwrapped_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        unwrapped_b64 = base64.b64encode(buf).decode('utf-8')

        # 결함 마스크 전개 + 오버레이 합성
        DEFECT_COLORS = {
            'rust': (60, 76, 231),      # BGR red
            'scale': (15, 196, 241),     # BGR yellow
            'default': (0, 200, 255),    # BGR cyan
        }
        unwrapped_defects = []
        overlay_img = unwrapped_bgr.copy()
        has_defects = False

        # 전체 전개도 면적 (면적비 계산용)
        unwrap_total_px = mapper.out_w * mapper.out_h
        mm_per_px = 1.0 / self.pixel_per_mm
        unwrap_total_mm2 = round(unwrap_total_px * (mm_per_px ** 2), 2)

        # 가시 영역 (프레임에서 실제 샘플링된 전개도 픽셀 = 검은색 제외)
        visible_mask_uw = np.any(unwrapped_bgr > 0, axis=-1)
        visible_unwrap_px = int(np.count_nonzero(visible_mask_uw))
        visible_unwrap_mm2 = round(visible_unwrap_px * (mm_per_px ** 2), 2)
        visible_coverage_pct = (round(visible_unwrap_px / unwrap_total_px * 100, 2)
                                 if unwrap_total_px else 0.0)

        # 깊이맵 + cos α — 결함이 있으면 무조건 계산(가중 면적비에 필요)
        depth_mm = None
        cos_alpha_map = None
        depth_heatmap_b64 = None
        depth_stats = None
        need_depth = include_depth_map or bool(defect_masks)
        if need_depth:
            depth_mm, cos_alpha_map = mapper.compute_depth_map(pose, frame_bgr.shape,
                                                                return_cos_alpha=True)

        # 물리 면적 변환용: 픽셀 1개의 물리 면적 = z² / (f² · cos α)  (mm²)
        # 전체 분모는 Unwrap과 동일한 π·D·L (mm²) — 두 방식 직접 비교 가능
        weight_map = None  # z² / cos α
        if depth_mm is not None and cos_alpha_map is not None:
            valid_d = np.isfinite(depth_mm) & np.isfinite(cos_alpha_map)
            cos_safe = np.where(valid_d, np.maximum(cos_alpha_map, 0.02), np.nan)  # floor: ≈ 88.9°
            weight_map = np.where(valid_d, (depth_mm ** 2) / cos_safe, 0.0)
        else:
            valid_d = None
        cylinder_section_mm2 = float(np.pi * self.pipe_diameter_mm * self.max_depth_mm)
        f_eff_sq = float(mapper.f) ** 2  # water 보정 포함된 유효 초점거리²

        if include_depth_map and depth_mm is not None:
            heat_bgr, (d_lo, d_hi) = mapper.render_depth_heatmap(depth_mm)
            heat_overlay = cv2.addWeighted(heat_bgr, 0.55, frame_bgr, 0.55, 0)
            _, dbuf = cv2.imencode('.jpg', heat_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            depth_heatmap_b64 = base64.b64encode(dbuf).decode('utf-8')
            dv = depth_mm[np.isfinite(depth_mm)]
            if dv.size > 0:
                depth_stats = {
                    'min_mm': round(float(np.min(dv)), 2),
                    'max_mm': round(float(np.max(dv)), 2),
                    'mean_mm': round(float(np.mean(dv)), 2),
                    'colormap_range_mm': [round(d_lo, 2), round(d_hi, 2)]
                                         if (d_lo is not None and d_hi is not None) else None,
                    'wall_coverage_pct': round(float(dv.size) /
                                                (depth_mm.shape[0] * depth_mm.shape[1]) * 100, 2),
                }

        if defect_masks:
            for dm in defect_masks:
                label = dm.get('label', 'unknown')
                mask = dm['mask']  # (H, W) uint8

                # 원본 프레임 상 마스크 통계 (거리 + z²/cos α 물리면적)
                src_area_px = int(np.count_nonzero(mask))
                avg_dist_mm = None
                weighted_ratio_pct = None
                weighted_area_mm2 = None
                if depth_mm is not None and src_area_px > 0:
                    mask_bool = mask > 0
                    sel = depth_mm[mask_bool]
                    sel = sel[np.isfinite(sel)]
                    if sel.size > 0:
                        avg_dist_mm = round(float(np.mean(sel)), 2)
                    # 물리 면적: Σ[z² / cos α] / f²  (cos α 보정 포함)
                    if weight_map is not None and valid_d is not None and cylinder_section_mm2 > 0:
                        poly_valid = mask_bool & valid_d
                        poly_w = float(weight_map[poly_valid].sum())
                        weighted_area_mm2 = round(poly_w / f_eff_sq, 2)
                        # Unwrap과 동일 분모 (π·D·L)로 정규화
                        weighted_ratio_pct = round(weighted_area_mm2 / cylinder_section_mm2 * 100, 4)

                mask_unwrapped = mapper.unwrap(mask, pose)
                mask_binary = (mask_unwrapped > 0).astype(np.uint8) * 255

                area_px = int(np.count_nonzero(mask_binary))
                area_mm2 = round(area_px * mm_per_px * mm_per_px, 2)

                # bbox + 채움 비율 (회전 무시 축정렬)
                bbox_w_mm = bbox_h_mm = aspect_wh = poly_fill_pct = None
                if area_px > 0:
                    ys_idx, xs_idx = np.where(mask_binary > 0)
                    bbox_w_px = int(xs_idx.max() - xs_idx.min() + 1)
                    bbox_h_px = int(ys_idx.max() - ys_idx.min() + 1)
                    bbox_w_mm = round(bbox_w_px * mm_per_px, 2)
                    bbox_h_mm = round(bbox_h_px * mm_per_px, 2)
                    if bbox_h_mm and bbox_h_mm > 0:
                        aspect_wh = round(bbox_w_mm / bbox_h_mm, 3)
                    bbox_px = bbox_w_px * bbox_h_px
                    if bbox_px > 0:
                        poly_fill_pct = round(area_px / bbox_px * 100, 2)
                area_ratio_pct = (round(area_px / unwrap_total_px * 100, 4)
                                  if unwrap_total_px else None)
                # 가시 영역 대비 면적비 (검정 영역 제외)
                area_ratio_visible_pct = (round(area_px / visible_unwrap_px * 100, 4)
                                           if visible_unwrap_px else None)

                # 색상 결정
                color = DEFECT_COLORS['default']
                for key, c in DEFECT_COLORS.items():
                    if key != 'default' and key in label.lower():
                        color = c
                        break

                # 오버레이 합성
                if area_px > 0:
                    has_defects = True
                    colored = np.zeros_like(overlay_img)
                    colored[mask_binary > 0] = color
                    cv2.addWeighted(colored, 0.4, overlay_img, 1.0, 0, overlay_img)
                    # 외곽선
                    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay_img, contours, -1, color, 2, cv2.LINE_AA)
                    # 라벨
                    if contours:
                        M = cv2.moments(contours[0])
                        if M['m00'] > 0:
                            cx_label = int(M['m10'] / M['m00'])
                            cy_label = int(M['m01'] / M['m00'])
                            text = f"{label} {area_mm2:.1f}mm2"
                            cv2.putText(overlay_img, text, (cx_label - 40, cy_label - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.putText(overlay_img, text, (cx_label - 40, cy_label - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

                unwrapped_defects.append({
                    'label': label,
                    'area_px': area_px,
                    'area_mm2': area_mm2,
                    'area_cm2': round(area_mm2 / 100.0, 4),
                    'area_ratio_pct': area_ratio_pct,              # Unwrap 면적비 (π·D·L 분모)
                    'area_ratio_visible_pct': area_ratio_visible_pct,  # 가시 영역 분모
                    'src_area_px': src_area_px,                    # 원본 프레임 픽셀 수
                    'src_ratio_pct': (round(src_area_px / (frame_bgr.shape[0] *
                                                           frame_bgr.shape[1]) * 100, 4)
                                      if src_area_px else 0.0),    # Naive Frame 면적비
                    'weighted_ratio_pct': weighted_ratio_pct,      # z² 가중 Frame 면적비 (π·D·L 분모)
                    'weighted_area_mm2': weighted_area_mm2,        # Σz²/cos α/f² 기반 물리 면적 추정
                    'avg_camera_distance_mm': avg_dist_mm,
                    'bbox_width_mm': bbox_w_mm,                    # 전개도 축정렬 bbox 폭(mm)
                    'bbox_height_mm': bbox_h_mm,                   # 전개도 축정렬 bbox 높이(mm)
                    'aspect_wh': aspect_wh,                        # w/h — 원래 직사각형 종횡비와 비교용
                    'polygon_fill_pct': poly_fill_pct,             # 폴리곤이 bbox 의 몇 % 차지 (회전 여부 힌트)
                })

        # 오버레이 이미지 인코딩
        unwrapped_overlay_b64 = None
        if has_defects:
            _, obuf = cv2.imencode('.jpg', overlay_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            unwrapped_overlay_b64 = base64.b64encode(obuf).decode('utf-8')

        coord = mapper.get_coordinate_system()

        return {
            'pose': {
                'vp_x': round(vp_x, 2),
                'vp_y': round(vp_y, 2),
                'angle': round(angle, 2),
                'step': round(step, 4),
                'raw': ppnet_result.get('raw'),
                'source': ppnet_result.get('pose_source', 'ppnet'),
            },
            'camera': {
                'resolution': res_key,
                'f': cam['f'],
                'cx': cam['cx'],
                'cy': cam['cy'],
                'water': self.water,
            },
            'unwrapped_rgb_b64': unwrapped_b64,
            'unwrapped_overlay_b64': unwrapped_overlay_b64,
            'unwrapped_width': mapper.out_w,
            'unwrapped_height': mapper.out_h,
            'unwrap_total_px': unwrap_total_px,
            'unwrap_total_mm2': unwrap_total_mm2,
            'visible_unwrap_px': visible_unwrap_px,
            'visible_unwrap_mm2': visible_unwrap_mm2,
            'visible_coverage_pct': visible_coverage_pct,
            'unwrapped_defects': unwrapped_defects,
            'coordinate_system': coord,
            'depth_heatmap_b64': depth_heatmap_b64,
            'depth_stats': depth_stats,
        }


# ═══════════════════════════════════════════
#  PerformanceEvaluator — MAPE 평가
# ═══════════════════════════════════════════
class PerformanceEvaluator:
    """Reference Marker(직경 19mm) 매핑 정확도 평가

    원본: Performance_indicator/util/Performance_Evaluation.py

    디렉터리 규칙:
        foo.png      — RGB 이미지
        foo_mask.png — Reference Marker 이진 마스크

    반환:
        per-sample 매핑된 마커 픽셀수, 기준 면적(px), MAPE(%), accuracy(%)
    """

    REFERENCE_MARKER_MM = 19.0

    def __init__(self, ppnet_model_path, pipe_type=None, pipe_diameter_mm=None,
                 water=False, pixel_per_mm=10.0, max_depth_mm=300):
        if pipe_type and pipe_type in PIPE_DIMENSIONS:
            pr = PIPE_DIMENSIONS[pipe_type]
            if pipe_diameter_mm is None:
                pipe_diameter_mm = pr['diameter_mm']
            if water is None:
                water = pr['water_default']
        if pipe_diameter_mm is None:
            raise ValueError("pipe_diameter_mm or a valid pipe_type is required")

        self.ppnet = PPNet(ppnet_model_path)
        self.pipe_diameter_mm = pipe_diameter_mm
        self.water = water
        self.pixel_per_mm = pixel_per_mm
        self.max_depth_mm = max_depth_mm
        self._mappers = {}

    def _get_mapper(self, res_key):
        if res_key not in self._mappers:
            cam = CAMERA_PARAMS[res_key]
            self._mappers[res_key] = PipeMapper(
                f=cam['f'], cx=cam['cx'], cy=cam['cy'],
                pipe_diameter_mm=self.pipe_diameter_mm,
                water=self.water,
                pixel_per_mm=self.pixel_per_mm,
                max_depth_mm=self.max_depth_mm,
            )
        return self._mappers[res_key]

    @staticmethod
    def collect_pairs(directory):
        """{name: (img_path, mask_path)} — foo.png + foo_mask.png"""
        if not os.path.isdir(directory):
            return []
        files = [f for f in os.listdir(directory) if f.lower().endswith('.png')]
        masks = {f for f in files if '_mask' in f}
        images = [f for f in files if '_mask' not in f]
        pairs = []
        for img in images:
            stem = os.path.splitext(img)[0]
            mname = f"{stem}_mask.png"
            if mname in masks:
                pairs.append((
                    os.path.splitext(img)[0],
                    os.path.join(directory, img),
                    os.path.join(directory, mname),
                ))
        return pairs

    def evaluate(self, directory):
        pairs = self.collect_pairs(directory)
        if not pairs:
            return {'pairs': 0, 'samples': [], 'mape_pct': None, 'accuracy_pct': None,
                    'reference_area_px': 0}

        samples = []
        predicted = []
        # GT 물리 면적 (mm²) — 19mm 마커의 이론적 원 면적
        reference_area_mm2 = float(np.pi * (self.REFERENCE_MARKER_MM / 2.0) ** 2)

        for name, img_path, mask_path in pairs:
            rgb_bgr = cv2.imread(img_path)
            mask_bgr = cv2.imread(mask_path)
            if rgb_bgr is None or mask_bgr is None:
                continue
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            res_key = detect_resolution(w, h)

            pose = self.ppnet.run(rgb)['pose']
            mapper = self._get_mapper(res_key)

            # ── ① Naive Frame: 원본 이미지 마스크 픽셀 카운트 (물리 변환 없음) ──
            mask_gray_src = (cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                             if mask_bgr.ndim == 3 else mask_bgr)
            mask_bool_src = mask_gray_src > 0
            naive_area_px = int(np.count_nonzero(mask_bool_src))
            naive_frame_ratio_pct = (round(naive_area_px / (h * w) * 100, 4)
                                     if (h * w) > 0 else 0.0)

            # ── ③ Unwrap ──
            mapped = mapper.unwrap(mask_bgr, pose)
            mapped_gray = (cv2.cvtColor(mapped, cv2.COLOR_BGR2GRAY)
                           if mapped.ndim == 3 else mapped)
            mapped_bin = np.where(mapped_gray > 0, 255, 0).astype(np.uint8)
            unwrap_area_px = int(np.count_nonzero(mapped_bin))
            unwrap_area_mm2 = unwrap_area_px * (1.0 / self.pixel_per_mm) ** 2
            unwrap_error_pct = round(abs(reference_area_mm2 - unwrap_area_mm2) /
                                      reference_area_mm2 * 100, 3)

            # ── ② Weighted Frame: Σ[z²/cos α]/f² (cos α 입사각 보정 포함) ──
            weighted_area_mm2 = None
            weighted_error_pct = None
            try:
                depth_mm_s, cos_alpha_s = mapper.compute_depth_map(pose, (h, w),
                                                                    return_cos_alpha=True)
                valid = np.isfinite(depth_mm_s) & np.isfinite(cos_alpha_s)
                if valid.any() and naive_area_px > 0:
                    cos_safe = np.where(valid, np.maximum(cos_alpha_s, 0.02), np.nan)
                    w_map = np.where(valid, (depth_mm_s ** 2) / cos_safe, 0.0)
                    poly_valid = mask_bool_src & valid
                    poly_w = float(w_map[poly_valid].sum())
                    f_eff_sq = float(mapper.f) ** 2
                    weighted_area_mm2 = poly_w / f_eff_sq
                    weighted_error_pct = round(abs(reference_area_mm2 - weighted_area_mm2) /
                                                reference_area_mm2 * 100, 3)
            except Exception:
                pass

            samples.append({
                'name': name,
                'resolution': res_key,
                'img_path': img_path,
                'mask_path': mask_path,
                'pose': {'vp_x': round(pose[0], 2), 'vp_y': round(pose[1], 2),
                         'angle': round(pose[2], 2), 'step': round(pose[3], 4)},

                # Naive
                'naive_area_px': naive_area_px,
                'naive_frame_ratio_pct': naive_frame_ratio_pct,

                # Weighted (mm² + error vs GT)
                'weighted_area_mm2': round(weighted_area_mm2, 2) if weighted_area_mm2 is not None else None,
                'weighted_error_pct': weighted_error_pct,

                # Unwrap (기존)
                'mapped_area_px': unwrap_area_px,
                'unwrap_area_mm2': round(unwrap_area_mm2, 2),
                'unwrap_error_pct': unwrap_error_pct,
            })
            predicted.append(unwrap_area_px)

        # 기준 면적: 반경 19/2 mm → pixel_per_mm 적용 원의 픽셀 수
        marker_radius_px = (self.REFERENCE_MARKER_MM / 2.0) * self.pixel_per_mm
        canvas = np.zeros((max(2, int(marker_radius_px * 4)),
                           max(2, int(marker_radius_px * 4))), dtype=np.uint8)
        cv2.circle(canvas,
                   (canvas.shape[1] // 2, canvas.shape[0] // 2),
                   int(round(marker_radius_px)), 255, -1)
        reference_area_px = int(np.count_nonzero(canvas))
        if reference_area_px == 0:
            reference_area_px = 28345  # 원본 fallback

        pred_arr = np.array(predicted, dtype=np.float64)
        mape = float(np.mean(np.abs(reference_area_px - pred_arr) / reference_area_px) * 100)

        # 전개도 전체 면적과 면적비 (하나의 PipeMapper = 고정)
        any_mapper = next(iter(self._mappers.values()), None)
        unwrap_total_px = (any_mapper.out_w * any_mapper.out_h) if any_mapper else 0
        reference_ratio_pct = (
            round(reference_area_px / unwrap_total_px * 100, 4)
            if unwrap_total_px else None
        )

        for s, p in zip(samples, predicted):
            err_pct = abs(reference_area_px - p) / reference_area_px * 100
            s['error_pct'] = round(err_pct, 3)  # legacy (unwrap error, same as unwrap_error_pct)
            s['mapped_ratio_pct'] = (round(p / unwrap_total_px * 100, 4)
                                     if unwrap_total_px else None)

        # ── 3-way 집계 ──
        def _stats(errors):
            arr = [e for e in errors if e is not None]
            if not arr:
                return None
            return {
                'mape_pct': round(float(np.mean(arr)), 3),
                'accuracy_pct': round(100 - float(np.mean(arr)), 3),
                'std_pct': round(float(np.std(arr)), 3),
                'max_error_pct': round(float(np.max(arr)), 3),
                'n': len(arr),
            }

        unwrap_stats = _stats([s['unwrap_error_pct'] for s in samples])
        weighted_stats = _stats([s['weighted_error_pct'] for s in samples])

        # Naive: GT로 변환 불가, 대신 프레임 대비 비율의 편차(거리 의존성 증거)
        naive_pxs = [s['naive_area_px'] for s in samples if s['naive_area_px'] is not None]
        naive_ratios = [s['naive_frame_ratio_pct'] for s in samples if s['naive_frame_ratio_pct'] is not None]
        naive_summary = None
        if naive_pxs:
            mn, mx = min(naive_pxs), max(naive_pxs)
            ratio_mn = min(naive_ratios) if naive_ratios else 0
            ratio_mx = max(naive_ratios) if naive_ratios else 0
            naive_summary = {
                'area_px_min': mn,
                'area_px_max': mx,
                'area_px_mean': round(float(np.mean(naive_pxs)), 1),
                'area_px_std': round(float(np.std(naive_pxs)), 1),
                'area_px_ratio_max_min': round(mx / mn, 2) if mn > 0 else None,
                'frame_ratio_min_pct': round(ratio_mn, 4),
                'frame_ratio_max_pct': round(ratio_mx, 4),
                'frame_ratio_ratio_max_min': round(ratio_mx / ratio_mn, 2) if ratio_mn > 0 else None,
                'n': len(naive_pxs),
            }

        return {
            'pairs': len(samples),
            'samples': samples,
            'reference_area_px': reference_area_px,
            'reference_area_mm2': round(reference_area_mm2, 2),
            'reference_marker_mm': self.REFERENCE_MARKER_MM,
            'reference_ratio_pct': reference_ratio_pct,
            'unwrap_total_px': unwrap_total_px,
            'pixel_per_mm': self.pixel_per_mm,
            'pipe_diameter_mm': self.pipe_diameter_mm,
            'water': self.water,
            # 3-way 집계
            'unwrap_summary': unwrap_stats,
            'weighted_summary': weighted_stats,
            'naive_summary': naive_summary,
            # legacy
            'mape_pct': round(mape, 3),
            'accuracy_pct': round(100 - mape, 3),
        }

    def evaluate_sample(self, img_path, mask_path):
        """단일 샘플 상세 결과 — 원본/마스크/전개도/오버레이 + 메트릭"""
        rgb_bgr = cv2.imread(img_path)
        mask_bgr = cv2.imread(mask_path)
        if rgb_bgr is None or mask_bgr is None:
            raise FileNotFoundError(f"cannot read {img_path} or {mask_path}")

        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        res_key = detect_resolution(w, h)

        pose_result = self.ppnet.run(rgb)
        pose = pose_result['pose']
        vp_x, vp_y, angle, step = pose

        mapper = self._get_mapper(res_key)
        mapped_bgr = mapper.unwrap(rgb_bgr, pose)  # BGR in → BGR-like ndarray
        mapped_mask = mapper.unwrap(mask_bgr, pose)

        if mapped_mask.ndim == 3:
            mapped_mask_gray = cv2.cvtColor(mapped_mask, cv2.COLOR_BGR2GRAY)
        else:
            mapped_mask_gray = mapped_mask
        mapped_bin = np.where(mapped_mask_gray > 0, 255, 0).astype(np.uint8)
        area_px = int(np.count_nonzero(mapped_bin))

        # ── GT 물리 면적 ──
        reference_area_mm2 = float(np.pi * (self.REFERENCE_MARKER_MM / 2.0) ** 2)
        marker_radius_px = (self.REFERENCE_MARKER_MM / 2.0) * self.pixel_per_mm
        ref_canvas_side = max(2, int(marker_radius_px * 4))
        ref_canvas = np.zeros((ref_canvas_side, ref_canvas_side), dtype=np.uint8)
        cv2.circle(ref_canvas, (ref_canvas_side // 2, ref_canvas_side // 2),
                   int(round(marker_radius_px)), 255, -1)
        reference_area_px = int(np.count_nonzero(ref_canvas))
        if reference_area_px == 0:
            reference_area_px = 28345

        error_pct = abs(reference_area_px - area_px) / reference_area_px * 100

        # ── ① Naive Frame 측정 ──
        mask_gray_src = (cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                         if mask_bgr.ndim == 3 else mask_bgr)
        mask_bool_src = mask_gray_src > 0
        naive_area_px = int(np.count_nonzero(mask_bool_src))
        naive_frame_ratio_pct = (round(naive_area_px / (h * w) * 100, 4)
                                 if (h * w) > 0 else 0.0)

        # ── ③ Unwrap 물리 면적 + GT 대비 오차 ──
        unwrap_area_mm2 = area_px * (1.0 / self.pixel_per_mm) ** 2
        unwrap_error_pct_mm2 = round(abs(reference_area_mm2 - unwrap_area_mm2) /
                                      reference_area_mm2 * 100, 3)

        # ── ② Weighted Frame (z²/cos α) + Depth Map ──
        depth_mm, cos_alpha_map = mapper.compute_depth_map(pose, (h, w),
                                                            return_cos_alpha=True)
        weighted_area_mm2 = None
        weighted_error_pct = None
        if depth_mm is not None and cos_alpha_map is not None:
            valid = np.isfinite(depth_mm) & np.isfinite(cos_alpha_map)
            if valid.any() and naive_area_px > 0:
                cos_safe = np.where(valid, np.maximum(cos_alpha_map, 0.02), np.nan)
                w_map = np.where(valid, (depth_mm ** 2) / cos_safe, 0.0)
                poly_valid = mask_bool_src & valid
                poly_w = float(w_map[poly_valid].sum())
                f_eff_sq = float(mapper.f) ** 2
                weighted_area_mm2 = round(poly_w / f_eff_sq, 2)
                weighted_error_pct = round(abs(reference_area_mm2 - weighted_area_mm2) /
                                            reference_area_mm2 * 100, 3)

        # Depth 히트맵 (원본 프레임 합성)
        depth_heatmap_b64 = None
        depth_stats = None
        try:
            heat_bgr, (d_lo, d_hi) = mapper.render_depth_heatmap(depth_mm)
            heat_overlay = cv2.addWeighted(heat_bgr, 0.55, rgb_bgr, 0.55, 0)
            _, hbuf = cv2.imencode('.jpg', heat_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            depth_heatmap_b64 = base64.b64encode(hbuf).decode('utf-8')
            dv = depth_mm[np.isfinite(depth_mm)]
            if dv.size > 0:
                depth_stats = {
                    'min_mm': round(float(np.min(dv)), 2),
                    'max_mm': round(float(np.max(dv)), 2),
                    'mean_mm': round(float(np.mean(dv)), 2),
                    'colormap_range_mm': [round(d_lo, 2), round(d_hi, 2)]
                                         if (d_lo is not None and d_hi is not None) else None,
                }
        except Exception:
            pass

        # 오버레이: 매핑된 마스크 외곽선(빨강) + 동일 centroid에 기준원(녹)
        overlay = mapped_bgr.copy()
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        mapped_color = np.zeros_like(overlay)
        mapped_color[mapped_bin > 0] = (40, 60, 230)  # BGR red
        cv2.addWeighted(mapped_color, 0.35, overlay, 1.0, 0, overlay)

        contours, _ = cv2.findContours(mapped_bin, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (40, 60, 230), 2, cv2.LINE_AA)

        # centroid
        cx = cy = None
        if contours:
            M = cv2.moments(contours[0])
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
        if cx is None:
            cx = overlay.shape[1] // 2
            cy = overlay.shape[0] // 2

        cv2.circle(overlay, (cx, cy), int(round(marker_radius_px)),
                   (80, 210, 80), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"mapped={area_px}px  ref={reference_area_px}px  err={error_pct:.2f}%",
                    (10, overlay.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"mapped={area_px}px  ref={reference_area_px}px  err={error_pct:.2f}%",
                    (10, overlay.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (40, 40, 40), 1, cv2.LINE_AA)

        # 원본 프레임 + VP + 마커 외곽선
        orig_overlay = rgb_bgr.copy()
        mask_gray_in = (cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
                        if mask_bgr.ndim == 3 else mask_bgr)
        mask_bin_in = (mask_gray_in > 0).astype(np.uint8) * 255
        src_contours, _ = cv2.findContours(mask_bin_in, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(orig_overlay, src_contours, -1, (40, 60, 230), 2, cv2.LINE_AA)
        vp_px = (int(round(vp_x)), int(round(vp_y)))
        cv2.circle(orig_overlay, vp_px, 14, (200, 120, 230), 2, cv2.LINE_AA)
        cv2.line(orig_overlay, (vp_px[0] - 20, vp_px[1]), (vp_px[0] + 20, vp_px[1]),
                 (200, 120, 230), 2, cv2.LINE_AA)
        cv2.line(orig_overlay, (vp_px[0], vp_px[1] - 20), (vp_px[0], vp_px[1] + 20),
                 (200, 120, 230), 2, cv2.LINE_AA)

        def _encode(img):
            _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return base64.b64encode(buf).decode('utf-8')

        unwrapped_mask_vis = cv2.cvtColor(mapped_bin, cv2.COLOR_GRAY2BGR)

        unwrap_total_px = mapper.out_w * mapper.out_h
        mm_per_px = 1.0 / self.pixel_per_mm
        unwrap_total_mm2 = unwrap_total_px * (mm_per_px ** 2)
        mapped_area_mm2 = area_px * (mm_per_px ** 2)
        ref_area_mm2_rasterized = reference_area_px * (mm_per_px ** 2)
        mapped_ratio_pct = (area_px / unwrap_total_px * 100) if unwrap_total_px else 0.0
        reference_ratio_pct = (reference_area_px / unwrap_total_px * 100) if unwrap_total_px else 0.0

        # 가시 영역 (RGB 전개도 픽셀 중 검정 제외)
        visible_mask_uw = np.any(mapped_bgr > 0, axis=-1)
        visible_unwrap_px = int(np.count_nonzero(visible_mask_uw))
        visible_coverage_pct = (round(visible_unwrap_px / unwrap_total_px * 100, 2)
                                 if unwrap_total_px else 0.0)
        mapped_ratio_visible_pct = (round(area_px / visible_unwrap_px * 100, 4)
                                     if visible_unwrap_px else None)

        return {
            'resolution': res_key,
            'pose': {
                'vp_x': round(vp_x, 2), 'vp_y': round(vp_y, 2),
                'angle': round(angle, 2), 'step': round(step, 4),
                'raw': pose_result['raw'],
            },
            'image_size': {'w': w, 'h': h},
            'unwrap_size': {'w': mapper.out_w, 'h': mapper.out_h},
            'unwrap_total_px': unwrap_total_px,
            'unwrap_total_mm2': round(unwrap_total_mm2, 2),
            'visible_unwrap_px': visible_unwrap_px,
            'visible_coverage_pct': visible_coverage_pct,
            # Unwrap 방식
            'mapped_area_px': area_px,
            'mapped_area_mm2': round(mapped_area_mm2, 2),
            'mapped_ratio_pct': round(mapped_ratio_pct, 4),
            'mapped_ratio_visible_pct': mapped_ratio_visible_pct,
            'unwrap_error_pct_mm2': unwrap_error_pct_mm2,
            # GT
            'reference_area_px': reference_area_px,
            'reference_area_mm2': round(reference_area_mm2, 2),         # π·(9.5)² = 283.53
            'reference_area_mm2_rasterized': round(ref_area_mm2_rasterized, 2),  # 래스터화 (28345 ÷ 100)
            'reference_ratio_pct': round(reference_ratio_pct, 4),
            'error_pct': round(error_pct, 3),                            # unwrap px 기반 error (legacy)
            # Naive 방식
            'naive_area_px': naive_area_px,
            'naive_frame_ratio_pct': naive_frame_ratio_pct,
            # Weighted 방식
            'weighted_area_mm2': weighted_area_mm2,
            'weighted_error_pct': weighted_error_pct,
            # 공통
            'pixel_per_mm': self.pixel_per_mm,
            'max_depth_mm': self.max_depth_mm,
            'original_overlay_b64': _encode(orig_overlay),
            'unwrapped_rgb_b64': _encode(mapped_bgr if mapped_bgr.ndim == 3
                                         else cv2.cvtColor(mapped_bgr, cv2.COLOR_GRAY2BGR)),
            'unwrapped_mask_b64': _encode(unwrapped_mask_vis),
            'unwrapped_overlay_b64': _encode(overlay),
            'depth_heatmap_b64': depth_heatmap_b64,
            'depth_stats': depth_stats,
            'pipe_diameter_mm': self.pipe_diameter_mm,
            'water': self.water,
        }
