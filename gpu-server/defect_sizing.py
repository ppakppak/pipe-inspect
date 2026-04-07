#!/usr/bin/env python3
"""
결함 크기/면적 산출 모듈 (Defect Size/Area Calculation)

소실점(VP) 탐지 + Depth Estimation + 관경 정보를 결합하여
pixel-to-mm 변환을 수행하고, 구간별 면적비를 산출한다.
"""

import cv2
import numpy as np
import math
import time
import json
import os
import base64
from io import BytesIO
from pathlib import Path
from datetime import datetime


def _unwrap_cyclic_x_coords(xs, width):
    """원통 seam(0°/360° 경계)을 넘는 X 좌표를 연속적으로 펼친다."""
    if not xs:
        return xs
    unwrapped = [float(xs[0])]
    half = float(width) / 2.0
    for x in xs[1:]:
        x = float(x)
        prev = unwrapped[-1]
        while x - prev > half:
            x -= float(width)
        while prev - x > half:
            x += float(width)
        unwrapped.append(x)
    return unwrapped


# ============================================================
# Phase 1: 소실점(Vanishing Point) 탐지
# ============================================================

class VanishingPointDetector:
    """파이프 내부 영상에서 소실점(VP)을 탐지한다.

    파이프 CCTV 영상 특성:
      - VP = 파이프가 깊어지며 어두워지는 수렴점
      - VP 주변에서 바깥으로 방사형 밝기 증가
      - 카메라 장착 위치에 따라 VP가 중앙에서 편향될 수 있음
      - 상단에 OSD 텍스트(노란 글씨)가 항상 존재

    방법 1 (primary): Radial Darkness Convergence — 방사형 밝기 수렴점
    방법 2 (secondary): Gaussian Darkest Region — 대블러 후 최소밝기점
    자동 선택: 두 방법의 confidence를 비교하여 더 높은 쪽 채택
    """

    # OSD 텍스트 영역 (상단 ~15%)
    OSD_MASK_RATIO = 0.15

    def __init__(self):
        self._cache = {}  # video_id -> {vp, timestamp}

    def detect(self, frame_rgb, video_id=None) -> dict:
        """프레임에서 소실점을 탐지한다."""
        # 캐시 확인
        if video_id and video_id in self._cache:
            cached = self._cache[video_id]
            if time.time() - cached['timestamp'] < 3600:
                return cached['vp']

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY) if len(frame_rgb.shape) == 3 else frame_rgb.copy()
        gray_masked = self._mask_osd(gray)

        # 방법 1: 방사형 밝기 수렴점
        result_radial = self._detect_radial_convergence(gray_masked)

        # 방법 2: Gaussian darkest region
        result_dark = self._detect_gaussian_darkest(gray_masked)

        # confidence 비교 → 더 높은 쪽 채택
        if result_radial['confidence'] >= result_dark['confidence']:
            result = result_radial
        else:
            result = result_dark

        # 캐싱
        if video_id:
            self._cache[video_id] = {
                'vp': result,
                'timestamp': time.time()
            }

        return result

    def detect_batch(self, frames, video_id=None) -> dict:
        """여러 프레임에서 VP를 탐지하고 중앙값을 반환한다."""
        results = []
        for frame in frames:
            r = self.detect(frame)
            if r['confidence'] >= 0.1:
                results.append(r)

        if not results:
            h, w = frames[0].shape[:2]
            return {
                'results': [],
                'median_vp': {'vp_x': w // 2, 'vp_y': h // 2, 'confidence': 0.0, 'method': 'center_fallback'}
            }

        vp_xs = [r['vp_x'] for r in results]
        vp_ys = [r['vp_y'] for r in results]
        median_vp = {
            'vp_x': float(np.median(vp_xs)),
            'vp_y': float(np.median(vp_ys)),
            'confidence': float(np.mean([r['confidence'] for r in results])),
            'method': 'batch_median'
        }

        if video_id:
            self._cache[video_id] = {
                'vp': median_vp,
                'timestamp': time.time()
            }

        return {
            'results': results,
            'median_vp': median_vp
        }

    # 가장자리 비네팅 마스킹 비율 (각 변에서 5%)
    EDGE_MASK_RATIO = 0.05

    def _mask_osd(self, gray):
        """상단 OSD 텍스트 영역 + 가장자리 비네팅을 마스킹한다."""
        h, w = gray.shape[:2]
        masked = gray.copy()

        # 상단 OSD 영역
        osd_h = int(h * self.OSD_MASK_RATIO)
        if osd_h > 0 and osd_h < h // 2:
            fill_val = int(gray[osd_h:osd_h * 2, :].mean())
            masked[:osd_h, :] = fill_val

        # 가장자리 비네팅 마스킹 (가장자리를 주변 평균으로 채움)
        em = self.EDGE_MASK_RATIO
        ey = max(int(h * em), 4)
        ex = max(int(w * em), 4)
        interior_mean = int(gray[osd_h + ey:h - ey, ex:w - ex].mean())
        masked[h - ey:, :] = interior_mean      # 하단
        masked[:, :ex] = interior_mean           # 좌측
        masked[:, w - ex:] = interior_mean       # 우측

        return masked

    def _detect_radial_convergence(self, gray) -> dict:
        """방사형 밝기 수렴점(Radial Darkness Convergence) 탐지.

        원리: 파이프 VP에서는 모든 방향으로 밝기가 증가한다.
        후보점마다 8방향 밝기 증가 정도를 점수화 → 최고점 = VP.
        """
        h, w = gray.shape[:2]

        # 큰 블러로 노이즈 제거 + 부드러운 밝기 맵 생성
        ksize = max(w, h) // 8
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        smooth = cv2.GaussianBlur(gray, (ksize, ksize), 0).astype(np.float64)

        # 후보점 격자 (OSD 영역 제외)
        osd_h = int(h * self.OSD_MASK_RATIO)
        step = max(w, h) // 30  # ~20-25 steps per axis
        step = max(step, 8)

        best_score = -1
        best_x, best_y = w // 2, h // 2

        # 8방향 단위벡터
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        dirs_x = np.cos(angles)
        dirs_y = np.sin(angles)

        # 방사형 샘플링 거리 (5단계)
        max_radius = min(w, h) * 0.4
        radii = np.linspace(max_radius * 0.1, max_radius, 5)

        for cy in range(osd_h + step, h - step, step):
            for cx in range(step, w - step, step):
                center_val = smooth[cy, cx]
                total_increase = 0
                valid_dirs = 0

                for dx, dy in zip(dirs_x, dirs_y):
                    increases = []
                    for r in radii:
                        sx = int(cx + dx * r)
                        sy = int(cy + dy * r)
                        if 0 <= sx < w and 0 <= sy < h:
                            increases.append(smooth[sy, sx] - center_val)

                    if increases:
                        # 이 방향에서 밝기가 일관되게 증가하는지
                        avg_increase = np.mean(increases)
                        if avg_increase > 0:
                            # monotonic 증가 보너스 (순서대로 증가할수록 높은 점수)
                            monotonic_bonus = 1.0
                            for k in range(1, len(increases)):
                                if increases[k] >= increases[k - 1]:
                                    monotonic_bonus += 0.3
                            total_increase += avg_increase * monotonic_bonus
                            valid_dirs += 1

                # 점수: 방향 커버리지 × 평균 밝기 증가량
                if valid_dirs >= 6:  # 최소 6/16 방향에서 증가
                    coverage = valid_dirs / len(angles)
                    score = total_increase * coverage
                    if score > best_score:
                        best_score = score
                        best_x, best_y = cx, cy

        # 서브픽셀 정밀화: 최적점 주변 ±step 범위에서 미세 탐색
        fine_step = max(step // 4, 2)
        fine_best_score = best_score
        fine_x, fine_y = best_x, best_y

        for cy in range(max(osd_h, best_y - step), min(h, best_y + step + 1), fine_step):
            for cx in range(max(0, best_x - step), min(w, best_x + step + 1), fine_step):
                center_val = smooth[cy, cx]
                total_increase = 0
                valid_dirs = 0

                for dx, dy in zip(dirs_x, dirs_y):
                    increases = []
                    for r in radii:
                        sx = int(cx + dx * r)
                        sy = int(cy + dy * r)
                        if 0 <= sx < w and 0 <= sy < h:
                            increases.append(smooth[sy, sx] - center_val)

                    if increases:
                        avg_increase = np.mean(increases)
                        if avg_increase > 0:
                            monotonic_bonus = 1.0
                            for k in range(1, len(increases)):
                                if increases[k] >= increases[k - 1]:
                                    monotonic_bonus += 0.3
                            total_increase += avg_increase * monotonic_bonus
                            valid_dirs += 1

                if valid_dirs >= 6:
                    coverage = valid_dirs / len(angles)
                    score = total_increase * coverage
                    if score > fine_best_score:
                        fine_best_score = score
                        fine_x, fine_y = cx, cy

        # Confidence 계산
        if fine_best_score <= 0:
            return {'vp_x': w // 2, 'vp_y': h // 2, 'confidence': 0.0, 'method': 'radial_failed'}

        # score 기반 confidence (경험적 정규화)
        # 실제 파이프 영상에서 score 600~900 관찰 → 이 범위를 0.5~0.9로 매핑
        confidence = min(1.0, max(0.1, fine_best_score / 800.0))

        return {
            'vp_x': float(fine_x),
            'vp_y': float(fine_y),
            'confidence': round(confidence, 3),
            'method': 'radial_convergence',
            'score': round(fine_best_score, 1)
        }

    def _detect_gaussian_darkest(self, gray) -> dict:
        """Gaussian 블러 후 가장 어두운 영역의 중심을 VP로 추정.

        큰 커널 블러로 국소 노이즈를 제거하고,
        가장 어두운 영역의 무게중심을 찾는다.
        """
        h, w = gray.shape[:2]

        # 큰 Gaussian 블러 (관경의 ~1/4 수준)
        ksize = max(w, h) // 4
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        ksize = max(ksize, 31)
        smooth = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # 최소값 찾기
        min_val = smooth.min()
        max_val = smooth.max()
        val_range = max_val - min_val

        if val_range < 5:
            return {'vp_x': w // 2, 'vp_y': h // 2, 'confidence': 0.0, 'method': 'dark_no_contrast'}

        # 최소값 + 10% 이내 영역의 무게중심
        threshold = min_val + val_range * 0.10
        dark_mask = (smooth <= threshold).astype(np.float32)

        # 무게중심 계산
        ys, xs = np.where(smooth <= threshold)
        if len(xs) == 0:
            return {'vp_x': w // 2, 'vp_y': h // 2, 'confidence': 0.0, 'method': 'dark_no_region'}

        # 밝기의 역수로 가중 (더 어두울수록 높은 가중치)
        weights = (threshold - smooth[ys, xs]).astype(np.float64)
        weights_sum = weights.sum()
        if weights_sum == 0:
            vp_x = float(xs.mean())
            vp_y = float(ys.mean())
        else:
            vp_x = float(np.sum(xs * weights) / weights_sum)
            vp_y = float(np.sum(ys * weights) / weights_sum)

        # Confidence: 어두운 영역이 얼마나 집중되어 있는지
        # spread가 작을수록 높은 confidence
        spread_x = np.std(xs)
        spread_y = np.std(ys)
        avg_spread = (spread_x + spread_y) / 2
        spread_ratio = avg_spread / max(w, h)
        concentration = max(0, 1.0 - spread_ratio * 4)  # spread가 25% 이상이면 0

        # 어두운 영역과 밝은 영역의 대비가 클수록 높은 confidence
        contrast_score = min(1.0, val_range / 100.0)

        confidence = concentration * 0.6 + contrast_score * 0.4

        # 방사형 검증: VP에서 바깥으로 밝기가 증가하는지 확인
        radial_verify = self._verify_radial_increase(smooth, vp_x, vp_y)
        confidence *= (0.5 + 0.5 * radial_verify)  # 검증 실패 시 50% 감소

        return {
            'vp_x': float(vp_x),
            'vp_y': float(vp_y),
            'confidence': float(round(min(1.0, confidence), 3)),
            'method': 'gaussian_darkest',
            'dark_region_size': int(len(xs)),
            'contrast': float(round(float(val_range), 1))
        }

    def _verify_radial_increase(self, smooth, vp_x, vp_y) -> float:
        """VP 후보점에서 방사형으로 밝기가 증가하는지 검증.
        Returns: 0~1 (1=완벽한 방사형 증가)
        """
        h, w = smooth.shape[:2]
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        max_r = min(w, h) * 0.35
        pass_count = 0

        for angle in angles:
            dx = math.cos(angle)
            dy = math.sin(angle)
            prev_val = float(smooth[int(min(max(vp_y, 0), h - 1)), int(min(max(vp_x, 0), w - 1))])
            increasing = True

            for r in np.linspace(max_r * 0.2, max_r, 4):
                sx = int(vp_x + dx * r)
                sy = int(vp_y + dy * r)
                if 0 <= sx < w and 0 <= sy < h:
                    val = float(smooth[sy, sx])
                    if val < prev_val - 3:  # 약간의 여유
                        increasing = False
                        break
                    prev_val = val

            if increasing:
                pass_count += 1

        return pass_count / len(angles)

    def generate_debug_image(self, frame, vp_result) -> np.ndarray:
        """VP 탐지 결과를 시각화한 디버그 이미지 생성."""
        debug = frame.copy()
        vp_x = int(vp_result['vp_x'])
        vp_y = int(vp_result['vp_y'])
        h, w = debug.shape[:2]

        # VP 십자선 (마젠타)
        cv2.line(debug, (vp_x, 0), (vp_x, h), (255, 0, 255), 2, cv2.LINE_AA)
        cv2.line(debug, (0, vp_y), (w, vp_y), (255, 0, 255), 2, cv2.LINE_AA)

        # VP 원
        cv2.circle(debug, (vp_x, vp_y), 10, (0, 255, 255), 3, cv2.LINE_AA)
        cv2.circle(debug, (vp_x, vp_y), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # 정보 텍스트
        conf = vp_result.get('confidence', 0)
        method = vp_result.get('method', 'unknown')
        text = f"VP({vp_x},{vp_y}) conf={conf:.2f} [{method}]"
        cv2.putText(debug, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return debug



# ============================================================
# Phase 2: Depth Estimation (MiDaS)
# ============================================================

class DepthEstimator:
    """MiDaS 기반 상대 깊이 추정.

    Lazy loading으로 첫 호출 시 모델 로드.
    """

    _instance = None
    _model = None
    _transform = None
    _device = None
    _model_type = None

    @classmethod
    def get_instance(cls, model_type='MiDaS_small'):
        """싱글톤 인스턴스 반환."""
        if cls._instance is None or cls._model_type != model_type:
            cls._instance = cls(model_type)
        return cls._instance

    def __init__(self, model_type='MiDaS_small'):
        self._model_type = model_type
        self._depth_cache = {}  # (video_id, frame_number) -> depth_map

    def _ensure_loaded(self):
        """모델이 로드되어 있는지 확인하고, 없으면 로드."""
        if DepthEstimator._model is not None and DepthEstimator._model_type == self._model_type:
            return

        import torch
        import ssl
        import os
        print(f"[DEPTH] Loading MiDaS model: {self._model_type}...")

        # SSL 인증서 경로 수정 (venv 환경에서 certifi CA 번들 사용)
        _orig_ssl_cert = os.environ.get('SSL_CERT_FILE')
        try:
            import certifi
            os.environ['SSL_CERT_FILE'] = certifi.where()
            print(f"[DEPTH] Using certifi CA bundle: {certifi.where()}")
        except ImportError:
            # certifi 없으면 SSL 검증 비활성화 (fallback)
            ssl._create_default_https_context = ssl._create_unverified_context
            print("[DEPTH] WARNING: certifi not available, disabling SSL verification")

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # MiDaS 모델 로드
            model = torch.hub.load("intel-isl/MiDaS", self._model_type, trust_repo=True)
            model.to(device)
            model.eval()

            # Transform 로드
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            if self._model_type == 'MiDaS_small':
                transform = midas_transforms.small_transform
            elif self._model_type in ('DPT_Large', 'DPT_Hybrid'):
                transform = midas_transforms.dpt_transform
            else:
                transform = midas_transforms.small_transform
        finally:
            # SSL 환경변수 복원
            if _orig_ssl_cert is not None:
                os.environ['SSL_CERT_FILE'] = _orig_ssl_cert
            elif 'SSL_CERT_FILE' in os.environ:
                del os.environ['SSL_CERT_FILE']

        DepthEstimator._model = model
        DepthEstimator._transform = transform
        DepthEstimator._device = device
        DepthEstimator._model_type = self._model_type

        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

        print(f"[DEPTH] MiDaS loaded on {device}, VRAM: {vram_mb:.1f}MB")

    def estimate(self, frame_rgb, video_id=None, frame_number=None) -> np.ndarray:
        """상대 깊이 맵 추정 (0~1 정규화).

        Args:
            frame_rgb: BGR 프레임 (numpy array)
            video_id: 캐싱용 비디오 ID
            frame_number: 캐싱용 프레임 번호

        Returns:
            depth_map: 0~1 정규화된 깊이 맵 (float32, H×W)
        """
        import torch

        # 캐시 확인
        cache_key = (video_id, frame_number) if video_id and frame_number is not None else None
        if cache_key and cache_key in self._depth_cache:
            return self._depth_cache[cache_key]

        self._ensure_loaded()

        # RGB로 변환
        if len(frame_rgb.shape) == 3 and frame_rgb.shape[2] == 3:
            rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame_rgb

        # Transform 적용
        input_batch = DepthEstimator._transform(rgb).to(DepthEstimator._device)

        # 추론
        with torch.no_grad():
            prediction = DepthEstimator._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        # 0~1 정규화
        d_min = depth_map.min()
        d_max = depth_map.max()
        if d_max - d_min > 0:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        depth_map = depth_map.astype(np.float32)

        # 캐시 저장 (최대 50개)
        if cache_key:
            if len(self._depth_cache) > 50:
                oldest = next(iter(self._depth_cache))
                del self._depth_cache[oldest]
            self._depth_cache[cache_key] = depth_map

        return depth_map

    def get_vram_usage(self):
        """현재 VRAM 사용량 반환 (MB)."""
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0

    @staticmethod
    def depth_to_colorized(depth_map) -> np.ndarray:
        """깊이 맵을 컬러맵으로 변환 (시각화용)."""
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        return colorized

    @staticmethod
    def depth_to_base64_png(depth_map) -> str:
        """깊이 맵을 16bit PNG base64 문자열로 변환."""
        depth_uint16 = (depth_map * 65535).astype(np.uint16)
        _, buf = cv2.imencode('.png', depth_uint16)
        return base64.b64encode(buf.tobytes()).decode('utf-8')


# ============================================================
# Phase 3: 실물 크기 계산 (Pipe Size Calibration)
# ============================================================

class PipeSizeCalibrator:
    """관경 정보와 VP/Depth를 결합하여 pixel→mm 스케일 변환.

    원리:
    - 파이프 직경(mm)은 알려져 있음 (예: 300mm)
    - VP에서 먼 위치(카메라에 가까운 곳)에서 파이프 단면이 더 크게 보임
    - depth 값을 이용하여 각 위치에서의 mm/pixel 비율을 추정
    """

    def __init__(self, pipe_diameter_mm, vp_x, vp_y, image_w, image_h):
        self.pipe_diameter_mm = pipe_diameter_mm
        self.vp_x = vp_x
        self.vp_y = vp_y
        self.image_w = image_w
        self.image_h = image_h

        # 이미지에서 파이프 단면의 pixel 직경 추정
        # VP에서 가장 먼 모서리까지의 거리 × 2 ≈ 최대 가시 직경
        max_dist = max(
            math.hypot(vp_x, vp_y),
            math.hypot(image_w - vp_x, vp_y),
            math.hypot(vp_x, image_h - vp_y),
            math.hypot(image_w - vp_x, image_h - vp_y)
        )
        # 파이프가 이미지의 대부분을 차지한다고 가정
        # 이미지 중심에서 가장자리까지 = 파이프 반경
        self.reference_pipe_radius_px = min(image_w, image_h) / 2.0
        self.base_scale = pipe_diameter_mm / (2.0 * self.reference_pipe_radius_px)  # mm/px at reference

    def compute_scale_at_depth(self, depth_value, point_x, point_y) -> float:
        """주어진 위치와 depth에서의 mm/pixel 스케일 팩터 계산.

        Args:
            depth_value: 0~1 정규화된 깊이값 (0=먼곳, 1=가까운곳)
            point_x, point_y: 측정 위치

        Returns:
            mm_per_pixel: 해당 위치에서의 mm/pixel 비율
        """
        # VP로부터의 거리 (정규화)
        dist_from_vp = math.hypot(point_x - self.vp_x, point_y - self.vp_y)
        max_dist = math.hypot(self.image_w / 2, self.image_h / 2)
        norm_dist = min(dist_from_vp / max_dist, 1.0) if max_dist > 0 else 0

        # 깊이 보정:
        # depth_value 높음(가까움) → 실제 단면이 크게 보임 → mm/px 작음
        # depth_value 낮음(먼곳) → 실제 단면이 작게 보임 → mm/px 큼
        # VP 근처는 먼 곳이므로 스케일 팩터가 큼
        depth_factor = 1.0
        if depth_value > 0.01:
            # 깊이에 반비례하여 스케일 조정
            depth_factor = 0.5 / max(depth_value, 0.1)
        else:
            depth_factor = 5.0  # VP 근처 (매우 먼 곳)

        # VP로부터의 거리 보정:
        # 가장자리(가까운 곳) → 관의 실제 크기에 가까움
        # 중심(VP 근처, 먼 곳) → 원근 축소됨
        distance_factor = max(0.2, norm_dist)

        mm_per_pixel = self.base_scale * depth_factor * distance_factor
        return mm_per_pixel

    def compute_scale_at_position(self, depth_map, point_x, point_y) -> float:
        """깊이 맵을 사용하여 특정 위치의 mm/pixel 스케일 계산."""
        h, w = depth_map.shape[:2]
        # 좌표 clamp
        px = max(0, min(int(point_x), w - 1))
        py = max(0, min(int(point_y), h - 1))
        depth_value = float(depth_map[py, px])
        return self.compute_scale_at_depth(depth_value, point_x, point_y)

    def measure_defect(self, polygon_points, depth_map=None) -> dict:
        """결함의 실물 크기를 계산한다.

        Args:
            polygon_points: [[x1,y1], [x2,y2], ...] 형태의 폴리곤 좌표
            depth_map: 0~1 정규화된 깊이 맵 (없으면 VP 거리 기반 추정)

        Returns:
            측정 결과 딕셔너리
        """
        pts = np.array(polygon_points, dtype=np.float32)
        if len(pts) < 3:
            return {'error': 'polygon must have at least 3 points'}

        # Pixel 단위 측정
        # Bounding box
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        pixel_width = float(x_max - x_min)
        pixel_height = float(y_max - y_min)

        # Pixel area (Shoelace formula)
        n = len(pts)
        pixel_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            pixel_area += pts[i][0] * pts[j][1]
            pixel_area -= pts[j][0] * pts[i][1]
        pixel_area = abs(pixel_area) / 2.0

        # 결함 중심점
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())

        # mm/pixel 스케일 계산
        if depth_map is not None:
            scale = self.compute_scale_at_position(depth_map, cx, cy)

            # 결함 영역의 평균 스케일 (더 정확한 면적 계산)
            scales = []
            for pt in pts:
                s = self.compute_scale_at_position(depth_map, pt[0], pt[1])
                scales.append(s)
            avg_scale = float(np.mean(scales))
            confidence = 0.85
        else:
            # 깊이 맵 없이 VP 거리 기반 추정
            dist_from_vp = math.hypot(cx - self.vp_x, cy - self.vp_y)
            max_dist = math.hypot(self.image_w / 2, self.image_h / 2)
            norm_dist = min(dist_from_vp / max_dist, 1.0) if max_dist > 0 else 0.5
            scale = self.base_scale / max(norm_dist, 0.2)
            avg_scale = scale
            confidence = 0.5  # 깊이 정보 없이는 낮은 신뢰도

        real_width_mm = float(pixel_width * scale)
        real_height_mm = float(pixel_height * scale)
        real_area_mm2 = float(pixel_area * (avg_scale ** 2))
        real_area_cm2 = real_area_mm2 / 100.0

        return {
            'pixel_area': round(float(pixel_area), 1),
            'pixel_width': round(float(pixel_width), 1),
            'pixel_height': round(float(pixel_height), 1),
            'real_width_mm': round(real_width_mm, 1),
            'real_height_mm': round(real_height_mm, 1),
            'real_area_mm2': round(real_area_mm2, 1),
            'real_area_cm2': round(real_area_cm2, 2),
            'scale_factor_mm_per_px': round(float(avg_scale), 4),
            'measurement_confidence': round(float(confidence), 2),
            'center_px': {'x': round(float(cx), 1), 'y': round(float(cy), 1)},
            'measured_at': datetime.now().isoformat()
        }


# ============================================================
# Phase 4: 면적비 산출 (Pipe Area Ratio)
# ============================================================

class PipeAreaRatioCalculator:
    """관 내면 대비 결함 면적비를 산출한다."""

    def calculate_section_ratio(self, pipe_diameter_mm, section_length_mm=6000,
                                defect_measurements=None) -> dict:
        """6m 구간 대비 결함 면적비 산출.

        Args:
            pipe_diameter_mm: 관 직경 (mm)
            section_length_mm: 구간 길이 (mm), 기본 6m
            defect_measurements: 결함 측정 결과 리스트

        Returns:
            면적비 결과 딕셔너리
        """
        if defect_measurements is None:
            defect_measurements = []

        # 전체 관 내면적 = π × 직경 × 길이 (원통 측면적)
        total_pipe_area_mm2 = math.pi * pipe_diameter_mm * section_length_mm
        total_pipe_area_cm2 = total_pipe_area_mm2 / 100.0
        total_pipe_area_m2 = total_pipe_area_mm2 / 1_000_000.0

        # 결함별 면적 집계
        total_defect_area_mm2 = 0.0
        by_class = {}

        for m in defect_measurements:
            area = float(m.get('real_area_mm2', 0))
            label = m.get('label', 'unknown')
            total_defect_area_mm2 += area

            if label not in by_class:
                by_class[label] = {'area_mm2': 0, 'area_cm2': 0, 'count': 0, 'ratio_percent': 0}
            by_class[label]['area_mm2'] += area
            by_class[label]['area_cm2'] += area / 100.0
            by_class[label]['count'] += 1

        # 비율 계산
        defect_ratio_percent = (total_defect_area_mm2 / total_pipe_area_mm2 * 100) if total_pipe_area_mm2 > 0 else 0

        for label in by_class:
            by_class[label]['ratio_percent'] = round(
                by_class[label]['area_mm2'] / total_pipe_area_mm2 * 100, 4
            ) if total_pipe_area_mm2 > 0 else 0
            by_class[label]['area_mm2'] = round(by_class[label]['area_mm2'], 1)
            by_class[label]['area_cm2'] = round(by_class[label]['area_cm2'], 2)

        return {
            'pipe_diameter_mm': pipe_diameter_mm,
            'section_length_mm': section_length_mm,
            'total_pipe_area_mm2': round(total_pipe_area_mm2, 1),
            'total_pipe_area_cm2': round(total_pipe_area_cm2, 1),
            'total_pipe_area_m2': round(total_pipe_area_m2, 4),
            'total_defect_area_mm2': round(total_defect_area_mm2, 1),
            'total_defect_area_cm2': round(total_defect_area_mm2 / 100.0, 2),
            'defect_ratio_percent': round(defect_ratio_percent, 4),
            'defect_count': len(defect_measurements),
            'by_class': by_class
        }


# ============================================================
# Phase 5: 파이프 전개도 (Pipe Unwrapping)
# ============================================================

class PipeUnwrapper:
    """VP 중심 극좌표 변환으로 파이프 전개도 생성.

    핵심 알고리즘:
      1. VP를 원점으로 극좌표 변환 (dx, dy → theta, r)
      2. theta → X축 (원주 방향)
      3. r → Y축 (깊이 방향) — 균일 스케일링
      4. cv2.remap() + BORDER_REPLICATE로 이미지 밖은 가장자리 픽셀로 채움
    """

    def __init__(self, vp_x, vp_y, image_w, image_h,
                 pipe_diameter_mm=300, output_width=800, output_height=600):
        self.vp_x = float(vp_x)
        self.vp_y = float(vp_y)
        self.image_w = image_w
        self.image_h = image_h
        self.pipe_diameter_mm = pipe_diameter_mm
        self.output_width = output_width
        self.output_height = output_height

        # max_radius: VP에서 이미지 꼭짓점까지 최대 거리 (전체 콘텐츠 포함)
        corners = np.array([
            [0, 0], [image_w, 0],
            [0, image_h], [image_w, image_h]
        ], dtype=np.float64)
        dists = np.hypot(corners[:, 0] - vp_x, corners[:, 1] - vp_y)
        self.max_radius = float(np.max(dists))
        self.min_radius = 5.0  # VP 바로 주변 제외

        # 스케일: mm per pixel
        circumference_mm = math.pi * pipe_diameter_mm
        self.mm_per_px_x = circumference_mm / output_width
        self.mm_per_px_y = pipe_diameter_mm * 3.0 / output_height

        # remap 테이블 (lazy build)
        self._map_x = None
        self._map_y = None

    def _build_remap_tables(self):
        """np.meshgrid로 remap 테이블 생성 (균일 스케일링)."""
        if self._map_x is not None:
            return

        out_y_idx = np.arange(self.output_height, dtype=np.float32)
        out_x_idx = np.arange(self.output_width, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(out_x_idx, out_y_idx)

        # theta: -π ~ π (전체 원주)
        theta = grid_x / self.output_width * 2.0 * np.pi - np.pi

        # r: 위(y=0) = max_r(가까움), 아래(y=max) = min_r(VP/먼 곳)
        r = self.min_radius + (self.max_radius - self.min_radius) * (1.0 - grid_y / self.output_height)

        self._map_x = (self.vp_x + r * np.cos(theta)).astype(np.float32)
        self._map_y = (self.vp_y + r * np.sin(theta)).astype(np.float32)

    def unwrap(self, frame):
        """프레임을 전개도로 변환.

        BORDER_REPLICATE: 이미지 밖은 가장자리 픽셀로 채움 (V자 검은갭 방지).
        """
        self._build_remap_tables()
        unwrapped = cv2.remap(
            frame, self._map_x, self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        return unwrapped

    def transform_point(self, x, y):
        """원본 좌표 → 전개도 좌표 변환.

        Y축: 위(0) = 파이프 가장자리(가까움), 아래(max) = VP(먼 곳)
        """
        dx = x - self.vp_x
        dy = y - self.vp_y
        theta = math.atan2(dy, dx)
        r = math.hypot(dx, dy)

        if r < self.min_radius or r > self.max_radius:
            return None

        out_x = (theta + math.pi) / (2.0 * math.pi) * self.output_width
        out_y = (1.0 - (r - self.min_radius) / (self.max_radius - self.min_radius)) * self.output_height

        out_x = max(0.0, min(self.output_width - 1.0, out_x))
        out_y = max(0.0, min(self.output_height - 1.0, out_y))
        return (out_x, out_y)

    def transform_polygon(self, polygon_flat):
        """COCO flat polygon [x1,y1,x2,y2,...] → 전개도 좌표로 변환."""
        result = []
        for i in range(0, len(polygon_flat), 2):
            pt = self.transform_point(polygon_flat[i], polygon_flat[i + 1])
            if pt is None:
                dx = polygon_flat[i] - self.vp_x
                dy = polygon_flat[i + 1] - self.vp_y
                theta = math.atan2(dy, dx)
                r = math.hypot(dx, dy)
                out_x = (theta + math.pi) / (2.0 * math.pi) * self.output_width
                out_y = 0.0 if r > self.max_radius else float(self.output_height - 1)
                result.extend([max(0.0, min(self.output_width - 1.0, out_x)), out_y])
            else:
                result.extend([pt[0], pt[1]])
        return result

    def calculate_unwrapped_area(self, polygon_flat_unwrapped):
        """전개도 좌표의 폴리곤 면적 계산 (Shoelace 공식).

        Args:
            polygon_flat_unwrapped: 전개도 좌표 [x1,y1,x2,y2,...]

        Returns:
            dict: area_px2, area_mm2, area_cm2
        """
        n = len(polygon_flat_unwrapped) // 2
        if n < 3:
            return {'area_px2': 0, 'area_mm2': 0, 'area_cm2': 0}

        # Shoelace formula
        xs = [polygon_flat_unwrapped[i * 2] for i in range(n)]
        ys = [polygon_flat_unwrapped[i * 2 + 1] for i in range(n)]

        # 원통 seam(좌우 경계) 넘는 폴리곤 보정
        xs = _unwrap_cyclic_x_coords(xs, self.output_width)

        area_px2 = 0
        for i in range(n):
            j = (i + 1) % n
            area_px2 += xs[i] * ys[j]
            area_px2 -= xs[j] * ys[i]
        area_px2 = abs(area_px2) / 2.0

        area_mm2 = area_px2 * self.mm_per_px_x * self.mm_per_px_y
        area_cm2 = area_mm2 / 100.0

        return {
            'area_px2': round(area_px2, 1),
            'area_mm2': round(area_mm2, 2),
            'area_cm2': round(area_cm2, 4)
        }

    def get_coordinate_system(self):
        """좌표계 메타데이터 반환."""
        return {
            'x_axis': '원주 방향 (circumferential)',
            'y_axis': '파이프 축 방향 (axial / depth)',
            'x_range_mm': round(math.pi * self.pipe_diameter_mm, 2),
            'y_range_mm': round(self.pipe_diameter_mm * 3.0, 2),
            'mm_per_px_x': round(self.mm_per_px_x, 4),
            'mm_per_px_y': round(self.mm_per_px_y, 4),
            'output_width': self.output_width,
            'output_height': self.output_height,
            'pipe_diameter_mm': self.pipe_diameter_mm
        }


class DepthAwarePipeUnwrapper:
    """VP 중심 극좌표 변환 + MiDaS depth 기반 Y축 스케일링.

    PipeUnwrapper와 동일한 극좌표 remap을 사용하되,
    Y축 mm/px을 depth map에서 행별로 실측하여 물리적으로 정확한 면적 산출.
    """

    def __init__(self, vp_x, vp_y, image_w, image_h,
                 pipe_diameter_mm=300, output_width=800, output_height=600):
        self.vp_x = float(vp_x)
        self.vp_y = float(vp_y)
        self.image_w = image_w
        self.image_h = image_h
        self.pipe_diameter_mm = pipe_diameter_mm
        self.output_width = output_width
        self.output_height = output_height

        # max_radius: VP에서 이미지 꼭짓점까지 최대 거리
        corners = np.array([
            [0, 0], [image_w, 0],
            [0, image_h], [image_w, image_h]
        ], dtype=np.float64)
        dists = np.hypot(corners[:, 0] - vp_x, corners[:, 1] - vp_y)
        self.max_radius = float(np.max(dists))
        self.min_radius = 5.0

        # X축 스케일 (원주 방향 — 기존과 동일)
        circumference_mm = math.pi * pipe_diameter_mm
        self.mm_per_px_x = circumference_mm / output_width

        # Y축 스케일: depth 기반으로 행별 계산 (set_depth_map 호출 전까지 균일 fallback)
        self.mm_per_px_y_uniform = pipe_diameter_mm * 3.0 / output_height
        self.mm_per_px_y_array = None  # np.ndarray[output_height] — set_depth_map 후 설정

        # X축 스케일: 전개도는 θ 균등 매핑이므로 원주 방향은 균일
        # (파이프 단면 arc length = R·dθ, θ와 무관하게 일정)

        # remap 테이블 (lazy build)
        self._map_x = None
        self._map_y = None

    def _build_remap_tables(self):
        """np.meshgrid로 remap 테이블 생성 (바닥 카메라 기하학 보정).

        카메라가 파이프 바닥에 위치할 때, 깊이 z에서 파이프 단면은
        VP 중심이 아닌 편향된 원으로 투영됨:
          - VP에서의 거리: d(α) = 2R_img·sin(α/2)
          - α=0(바닥/카메라): d=0 (VP에 가까움)
          - α=π(상단): d=2R_img (VP에서 멀리)

        보정: 각 각도 θ에서 기준 r을 sin(α/2) 비율로 조정하여
        동일 물리 깊이가 동일 행에 매핑되도록 함.
        """
        if self._map_x is not None:
            return

        out_y_idx = np.arange(self.output_height, dtype=np.float32)
        out_x_idx = np.arange(self.output_width, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(out_x_idx, out_y_idx)

        # theta: -π ~ π (전체 원주)
        theta = grid_x / self.output_width * 2.0 * np.pi - np.pi

        # r_base: 위(y=0) = min_r(VP/먼 곳), 아래(y=max) = max_r(가까움)
        r_base = self.min_radius + (self.max_radius - self.min_radius) * (grid_y / self.output_height)

        # 바닥 카메라 기하학 보정
        # 카메라 위치: atan2 기준 π/2 (이미지 하단 방향)
        theta_cam = np.pi / 2.0

        # 카메라로부터의 원주각 α
        alpha = np.abs(theta - theta_cam)
        alpha = np.where(alpha > np.pi, 2.0 * np.pi - alpha, alpha)

        # VP에서의 거리 비율: sin(α/2)
        # α=0(바닥) → 0, α=π(상단) → 1
        # 평균값으로 정규화하여 전체 스케일 보존
        dist_ratio = np.sin(alpha / 2.0)
        mean_ratio = np.mean(dist_ratio)

        # 보정 강도: 파이프 반경의 이미지상 크기 비율
        # R_pipe_pixels / max_radius ≈ 파이프가 이미지에서 차지하는 비율
        R_pipe_pixels = min(self.image_w, self.image_h) / 2.0
        correction_strength = np.clip(R_pipe_pixels / self.max_radius, 0.0, 0.8)

        # 보정된 r: r_base에 각도별 편향 적용
        # 바닥(α=0): r 증가 (카메라 가까이 → 크게 보임 → 더 넓게 샘플링)
        # 상단(α=π): r 감소 (카메라 멀리 → 작게 보임 → 좁게 샘플링)
        r_correction = 1.0 + correction_strength * (1.0 - dist_ratio / mean_ratio)
        r = r_base * r_correction

        # 최소/최대 반경 클램핑
        r = np.clip(r, self.min_radius, self.max_radius * 1.5)

        self._map_x = (self.vp_x + r * np.cos(theta)).astype(np.float32)
        self._map_y = (self.vp_y + r * np.sin(theta)).astype(np.float32)

    def set_depth_map(self, depth_map, calibrator):
        """Depth map에서 행별 Y축 mm/px 스케일 계산 (바닥 카메라 기하학 보정).

        카메라가 파이프 바닥에 위치하므로:
        - 바닥(α≈0): depth ≈ 순수 축방향, 신뢰도 높음
        - 상단(α≈π): depth에 단면 거리 혼합, 신뢰도 낮음
        cos²(α/2) 가중치로 바닥 각도를 우선하여 Y스케일 정확도 향상.

        Args:
            depth_map: 0~1 정규화된 깊이 맵 (H×W float32)
            calibrator: PipeSizeCalibrator 인스턴스
        """
        h, w = depth_map.shape[:2]
        n_angles = 36
        angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)

        # 카메라 위치 = 바닥 = atan2 기준 π/2 (이미지 하단 방향)
        theta_cam = np.pi / 2.0

        # 좌표계 보정: 전개도 1px(Y) = 원본 radial_step px(반경방향)
        # calibrator는 원본 이미지 mm/px를 반환하므로 이 비율을 곱해야 함
        radial_step = (self.max_radius - self.min_radius) / self.output_height

        self.mm_per_px_y_array = np.zeros(self.output_height, dtype=np.float64)
        valid_mask = np.zeros(self.output_height, dtype=bool)

        for out_y in range(self.output_height):
            # 해당 행의 반경
            r = self.min_radius + (self.max_radius - self.min_radius) * (out_y / self.output_height)

            weighted_scales = []
            weights = []
            for theta in angles:
                src_x = self.vp_x + r * math.cos(theta)
                src_y = self.vp_y + r * math.sin(theta)

                # 이미지 범위 클리핑
                ix = int(np.clip(src_x, 0, w - 1))
                iy = int(np.clip(src_y, 0, h - 1))

                depth_val = float(depth_map[iy, ix])

                # 카메라로부터의 원주각 α 계산
                alpha = abs(theta - theta_cam)
                if alpha > np.pi:
                    alpha = 2.0 * np.pi - alpha

                # 가중치: 바닥(α≈0)→1.0, 옆면(α=π/2)→0.5, 상단(α=π)→0.0
                w_angle = math.cos(alpha / 2.0) ** 2

                if depth_val > 0.01:
                    scale = calibrator.compute_scale_at_depth(depth_val, src_x, src_y)
                    weighted_scales.append(scale)
                    weights.append(w_angle)

            if weighted_scales:
                # 가중 평균 (바닥 각도 우선) × radial_step 보정
                total_w = sum(weights)
                original_scale = sum(
                    s * w for s, w in zip(weighted_scales, weights)
                ) / total_w
                self.mm_per_px_y_array[out_y] = original_scale * radial_step
                valid_mask[out_y] = True

        # Fallback: 유효하지 않은 행은 가장 가까운 유효 행에서 보간
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            for out_y in range(self.output_height):
                if not valid_mask[out_y]:
                    # 가장 가까운 유효 행 찾기
                    nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - out_y))]
                    self.mm_per_px_y_array[out_y] = self.mm_per_px_y_array[nearest_idx]
        else:
            # 전체 fallback
            self.mm_per_px_y_array[:] = self.mm_per_px_y_uniform

        import sys
        print(f"[DepthUnwrap] radial_step={radial_step:.2f}, "
              f"valid_rows={valid_mask.sum()}/{self.output_height}, "
              f"Y스케일 min={self.mm_per_px_y_array.min():.4f}, "
              f"max={self.mm_per_px_y_array.max():.4f}, "
              f"mean={self.mm_per_px_y_array.mean():.4f}, "
              f"sum(Y범위)={self.mm_per_px_y_array.sum():.2f}mm",
              flush=True)

    def unwrap(self, frame):
        """프레임을 전개도로 변환."""
        self._build_remap_tables()
        return cv2.remap(
            frame, self._map_x, self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

    def unwrap_depth(self, depth_map):
        """Depth map을 같은 remap으로 전개 (시각화용)."""
        self._build_remap_tables()
        unwrapped_depth = cv2.remap(
            depth_map, self._map_x, self._map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        # MAGMA 컬러맵 적용
        depth_uint8 = (np.clip(unwrapped_depth, 0, 1) * 255).astype(np.uint8)
        colorized = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
        return colorized

    def _compute_r_correction(self, theta):
        """각도 θ에서의 r 보정 계수 계산 (바닥 카메라 기하학)."""
        theta_cam = math.pi / 2.0
        alpha = abs(theta - theta_cam)
        if alpha > math.pi:
            alpha = 2.0 * math.pi - alpha

        dist_ratio = math.sin(alpha / 2.0)
        mean_ratio = 2.0 / math.pi

        R_pipe_pixels = min(self.image_w, self.image_h) / 2.0
        correction_strength = min(R_pipe_pixels / self.max_radius, 0.8)

        return 1.0 + correction_strength * (1.0 - dist_ratio / mean_ratio)

    def transform_point(self, x, y):
        """원본 좌표 → 전개도 좌표 변환 (바닥 카메라 보정)."""
        dx = x - self.vp_x
        dy = y - self.vp_y
        theta = math.atan2(dy, dx)
        r = math.hypot(dx, dy)

        if r < self.min_radius or r > self.max_radius * 1.5:
            return None

        # 역보정: remap에서 r_corrected = r_base * correction 이므로
        # 원본 r에서 r_base(전개도 r)를 역산: r_base = r / correction
        r_correction = self._compute_r_correction(theta)
        r_base = r / r_correction if r_correction > 0.01 else r

        # r_base 범위 체크
        if r_base < self.min_radius or r_base > self.max_radius:
            # 범위 밖이면 클램핑
            r_base = max(self.min_radius, min(self.max_radius, r_base))

        out_x = (theta + math.pi) / (2.0 * math.pi) * self.output_width
        out_y = ((r_base - self.min_radius) / (self.max_radius - self.min_radius)) * self.output_height

        out_x = max(0.0, min(self.output_width - 1.0, out_x))
        out_y = max(0.0, min(self.output_height - 1.0, out_y))
        return (out_x, out_y)

    def transform_polygon(self, polygon_flat):
        """COCO flat polygon [x1,y1,x2,y2,...] → 전개도 좌표로 변환."""
        result = []
        for i in range(0, len(polygon_flat), 2):
            pt = self.transform_point(polygon_flat[i], polygon_flat[i + 1])
            if pt is None:
                dx = polygon_flat[i] - self.vp_x
                dy = polygon_flat[i + 1] - self.vp_y
                theta = math.atan2(dy, dx)
                r = math.hypot(dx, dy)
                r_correction = self._compute_r_correction(theta)
                r_base = r / r_correction if r_correction > 0.01 else r
                out_x = (theta + math.pi) / (2.0 * math.pi) * self.output_width
                out_y = 0.0 if r_base > self.max_radius else float(self.output_height - 1)
                result.extend([max(0.0, min(self.output_width - 1.0, out_x)), out_y])
            else:
                result.extend([pt[0], pt[1]])
        return result

    def calculate_unwrapped_area(self, polygon_flat_unwrapped):
        """Depth 보정 면적 계산 (행별 mm_per_px_y 적용 Shoelace).

        Returns:
            dict: area_px2, area_mm2, area_cm2, depth_corrected (bool)
        """
        n = len(polygon_flat_unwrapped) // 2
        if n < 3:
            return {'area_px2': 0, 'area_mm2': 0, 'area_cm2': 0, 'depth_corrected': False}

        xs = [polygon_flat_unwrapped[i * 2] for i in range(n)]
        ys = [polygon_flat_unwrapped[i * 2 + 1] for i in range(n)]

        # 원통 seam(좌우 경계) 넘는 폴리곤 보정
        xs = _unwrap_cyclic_x_coords(xs, self.output_width)

        if self.mm_per_px_y_array is not None:
            # 누적 Y스케일 테이블: pixel Y → physical Y (mm)
            # cumulative_y_mm[row] = Σ mm_per_px_y[0..row-1]
            cumulative_y_mm = np.zeros(self.output_height + 1, dtype=np.float64)
            for row in range(self.output_height):
                cumulative_y_mm[row + 1] = cumulative_y_mm[row] + self.mm_per_px_y_array[row]

            # 꼭짓점을 물리 좌표(mm)로 변환 후 Shoelace
            xs_mm = [x * self.mm_per_px_x for x in xs]
            ys_mm = []
            for y in ys:
                row_f = max(0.0, min(float(self.output_height - 1), y))
                row_lo = int(row_f)
                row_hi = min(row_lo + 1, self.output_height)
                frac = row_f - row_lo
                y_mm = cumulative_y_mm[row_lo] * (1 - frac) + cumulative_y_mm[row_hi] * frac
                ys_mm.append(y_mm)

            # Shoelace on physical coordinates
            area_mm2 = 0.0
            for i in range(n):
                j = (i + 1) % n
                area_mm2 += xs_mm[i] * ys_mm[j] - xs_mm[j] * ys_mm[i]
            area_mm2 = abs(area_mm2) / 2.0

            # pixel 면적
            area_px2 = 0.0
            for i in range(n):
                j = (i + 1) % n
                area_px2 += xs[i] * ys[j] - xs[j] * ys[i]
            area_px2 = abs(area_px2) / 2.0

            print(f"[Area] px2={area_px2:.1f}, area_mm2={area_mm2:.2f}, "
                  f"area_cm2={area_mm2/100:.4f}",
                  flush=True)

            return {
                'area_px2': round(area_px2, 1),
                'area_mm2': round(area_mm2, 2),
                'area_cm2': round(area_mm2 / 100.0, 4),
                'depth_corrected': True
            }
        else:
            # Fallback: 균일 스케일
            area_px2 = 0
            for i in range(n):
                j = (i + 1) % n
                area_px2 += xs[i] * ys[j] - xs[j] * ys[i]
            area_px2 = abs(area_px2) / 2.0

            area_mm2 = area_px2 * self.mm_per_px_x * self.mm_per_px_y_uniform
            return {
                'area_px2': round(area_px2, 1),
                'area_mm2': round(area_mm2, 2),
                'area_cm2': round(area_mm2 / 100.0, 4),
                'depth_corrected': False
            }

    def get_coordinate_system(self):
        """좌표계 메타데이터 반환 (Y스케일 통계 포함)."""
        result = {
            'x_axis': '원주 방향 (circumferential)',
            'y_axis': '파이프 축 방향 (axial / depth-corrected)',
            'x_range_mm': round(math.pi * self.pipe_diameter_mm, 2),
            'mm_per_px_x': round(self.mm_per_px_x, 4),
            'output_width': self.output_width,
            'output_height': self.output_height,
            'pipe_diameter_mm': self.pipe_diameter_mm
        }

        if self.mm_per_px_y_array is not None:
            y_scales = self.mm_per_px_y_array
            result['mm_per_px_y_min'] = round(float(np.min(y_scales)), 4)
            result['mm_per_px_y_max'] = round(float(np.max(y_scales)), 4)
            result['mm_per_px_y_mean'] = round(float(np.mean(y_scales)), 4)
            result['y_range_mm'] = round(float(np.sum(y_scales)), 2)
            result['depth_corrected'] = True
        else:
            result['mm_per_px_y'] = round(self.mm_per_px_y_uniform, 4)
            result['y_range_mm'] = round(self.pipe_diameter_mm * 3.0, 2)
            result['depth_corrected'] = False

        result['camera_position'] = 'bottom'

        return result


# ============================================================
# Phase 6: 결과 저장/로드
# ============================================================

class SizingResultManager:
    """크기 산출 결과를 파일로 저장/로드."""

    @staticmethod
    def get_results_dir(project_dir, video_id):
        """결과 저장 디렉토리 경로."""
        path = Path(project_dir) / 'sizing_results' / video_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def save_results(project_dir, video_id, results):
        """크기 산출 결과 저장."""
        results_dir = SizingResultManager.get_results_dir(project_dir, video_id)
        results_file = results_dir / 'sizing_results.json'

        # 기존 결과 로드
        existing = {}
        if results_file.exists():
            try:
                with open(results_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing = {}  # 깨진 파일은 무시

        # 업데이트
        existing.update(results)
        existing['updated_at'] = datetime.now().isoformat()

        # numpy 타입을 Python 기본 타입으로 변환하는 인코더
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                if hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                return super().default(obj)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        return str(results_file)

    @staticmethod
    def load_results(project_dir, video_id):
        """저장된 크기 산출 결과 로드."""
        results_dir = SizingResultManager.get_results_dir(project_dir, video_id)
        results_file = results_dir / 'sizing_results.json'

        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


# ============================================================
# 유틸리티 함수
# ============================================================

def frame_to_base64_jpeg(frame, quality=80) -> str:
    """프레임을 base64 JPEG 문자열로 변환."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buf = cv2.imencode('.jpg', frame, encode_params)
    return base64.b64encode(buf.tobytes()).decode('utf-8')
