#!/usr/bin/env python3
"""
관로 조사 통합 모듈 (Pipe Survey Analyzer)

SSIM 기반 정지/이동 감지 → 정지 구간만 YOLO Instance Segmentation +
PipeUnwrapper 극좌표 전개 → 연속 전개도 + Stop별 결함 분포를 생성한다.

핵심 원리:
  - 관내시경 카메라는 수동 조작 (밀고/끌기)
  - 정지 구간 = 작업자가 관심을 가진 지점 → 선명, 분석 가치 높음
  - 이동 구간 = 부유물, 흔들림 → 노이즈, 분석 가치 없음
  - 관벽 ROI의 SSIM으로 정지/이동 판별 (부유물에 강건)

출력:
  - 정지 구간(stop) 목록 + stop별 YOLO 결과
  - 연속 전개도 이미지 (정지 구간만, 결함 오버레이)
  - 프레임별 YOLO 결과 (비디오 오버레이용)
  - Stop별 결함 통계 JSON
"""

import cv2
import math
import numpy as np
import json
import os
import time
import requests
import base64
from pathlib import Path
from skimage.metrics import structural_similarity as ssim


# ─── 색상 팔레트 (클래스별, 부분 문자열 매칭) ───
DEFECT_COLORS_BGR = {
    'rust':   (60, 76, 231),    # #E74C3C (BGR)
    'scale':  (15, 196, 241),   # #F1C40F (BGR)
    'nodule': (0, 140, 255),    # 주황: corrosion_nodule
    'peel':   (255, 80, 80),    # coating_peel / peeling_silcoat
}
DEFAULT_COLOR_BGR = (255, 128, 0)  # cyan fallback


def _get_defect_color(class_name, colors=None):
    palette = colors if colors is not None else DEFECT_COLORS_BGR
    lower = class_name.lower()
    for key, color in palette.items():
        if key in lower:
            return color
    return DEFAULT_COLOR_BGR


# ─── YOLO 결과 → detections 변환 (api.py /api/survey/infer와 공유) ───

def yolo_result_to_detections(result, width, height,
                              polygon_scale=0.25, max_polygon_points=150):
    """Ultralytics 추론 결과 1건 → detections 리스트.

    각 detection: {box: [x,y,w,h], label, class_id, confidence,
                   area (마스크 있으면 마스크 픽셀 수), polygon: [[x,y],...]}
    """
    detections = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
        x1, y1, x2, y2 = box.astype(int)
        class_name = result.names[cls] if cls < len(result.names) else f'class_{cls}'
        area = int((x2 - x1) * (y2 - y1))

        det = {
            'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            'label': class_name,
            'class_id': int(cls),
            'confidence': round(float(conf), 3),
            'area': area,
        }

        # 마스크 → 폴리곤 추출 (다운스케일 후 contour, 최대 max_polygon_points점)
        if masks is not None and i < len(masks):
            mask = masks[i]
            small_w = int(width * polygon_scale)
            small_h = int(height * polygon_scale)
            mask_small = cv2.resize(mask, (small_w, small_h))
            mask_binary = (mask_small > 0.5).astype(np.uint8) * 255

            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                epsilon = 0.003 * cv2.arcLength(largest, True)
                approx = cv2.approxPolyDP(largest, epsilon, True)
                polygon_points = (approx.reshape(-1, 2) / polygon_scale).astype(int)
                if len(polygon_points) > max_polygon_points:
                    indices = np.linspace(0, len(polygon_points) - 1, max_polygon_points, dtype=int)
                    polygon_points = polygon_points[indices]
                det['polygon'] = polygon_points.tolist()

                # 마스크 기반 면적 (bbox보다 정확)
                mask_full = cv2.resize(mask, (width, height))
                det['area'] = int(np.sum(mask_full > 0.5))

        detections.append(det)

    return detections


def make_local_yolo_infer(model, conf=0.15, imgsz=640, device=None):
    """로컬 YOLO 모델로 동작하는 infer_fn 생성 (GPU 서버 없이 사용).

    PipeSurveyAnalyzer(infer_fn=make_local_yolo_infer(model)) 형태로 주입.
    """
    def infer_fn(frame_bgr):
        h, w = frame_bgr.shape[:2]
        kwargs = {'verbose': False, 'conf': conf, 'imgsz': imgsz}
        if device is not None:
            kwargs['device'] = device
        results = model.predict(frame_bgr, **kwargs)
        return yolo_result_to_detections(results[0], w, h)
    return infer_fn


class PipeSurveyAnalyzer:
    """정지 구간 기반으로 영상을 분석하여 연속 전개도 + 결함 분포를 생성한다."""

    def __init__(self, gpu=True, gpu_server_url='http://localhost:5004',
                 infer_fn=None, distance_fn=None, colors=None, preprocess_fn=None,
                 unwrap_fn=None):
        """
        Args:
            gpu_server_url: infer_fn 미지정 시 사용할 GPU 서버 주소 (HTTP 경로)
            infer_fn: fn(frame_bgr) -> detections 리스트. 지정하면 GPU 서버 없이
                      로컬 추론으로 동작 (make_local_yolo_infer 참고)
            distance_fn: fn(frame_bgr) -> float|None. 지정하면 stop별 대표 프레임에서
                         OSD 거리(m)를 읽어 stop['distance_m']에 기록
            colors: 결함 클래스 색상 팔레트 {부분문자열: BGR}. 미지정 시 모듈 기본값
            preprocess_fn: fn(frame_bgr) -> frame_bgr. 대표 프레임 전처리(렌즈 왜곡 보정 등)
            unwrap_fn: fn(frame_bgr, detections, pipe_diameter_mm) -> dict|None.
                       지정하면 극좌표 전개 대신 PPNet 기반 전개(GNU Mapping) 사용.
                       반환 dict: {visible_mask(HxW bool θ×z), defect_mask(HxW bool),
                                   pixel_per_mm(float), unwrapped_bgr, overlay_bgr(옵션)}
        """
        self.gpu_server_url = gpu_server_url
        self.infer_fn = infer_fn
        self.distance_fn = distance_fn
        self.colors = colors
        self.preprocess_fn = preprocess_fn
        self.unwrap_fn = unwrap_fn

    # ════════════════════════════════════════════
    #  메인 분석
    # ════════════════════════════════════════════
    def analyze_video(self, video_path, pipe_diameter_mm=300,
                      ssim_threshold=0.92, min_stop_frames=15,
                      scan_every_n=25,
                      strip_axial_mm=None, global_px_per_mm=0.2,
                      progress_callback=None) -> dict:
        """영상 전체 분석

        Phase 1: Motion Detection — SSIM 기반 정지/이동 구간 탐지
        Phase 2: 정지 구간별 YOLO 추론 + PipeUnwrapper 전개
        Phase 3: 후처리 — 전개도 결합, 결과 집계

        Args:
            video_path: 영상 파일 경로
            pipe_diameter_mm: 관경 (mm)
            ssim_threshold: 정지 판별 SSIM 임계값 (기본 0.92)
            min_stop_frames: 최소 정지 프레임 수 (기본 15 ≈ 0.6초@25fps)
            scan_every_n: Motion scan 간격 (기본 5프레임)
            progress_callback: fn(current, total, message)

        Returns:
            {
              video_path, pipe_diameter_mm, fps, total_frames,
              stops: [{index, start_frame, end_frame, duration_sec,
                       best_frame, detections, defects, ...}],
              motion_profile: [{frame, ssim_score}, ...],
              frame_results: [{frame_number, timestamp_sec, detections}, ...],
              summary: {...},
              panorama_path, panorama_overlay_path, stripmap_path,
              coordinate_system, vp,
            }
        """
        from defect_sizing import PipeUnwrapper

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0:
            fps = 25.0

        # ═══ Phase 1: VP 탐지 + Motion Detection ═══
        if progress_callback:
            progress_callback(0, 100, "Phase 1: 영상 스캔 중...")

        cap = cv2.VideoCapture(video_path)

        # VP 탐지 (초반 5프레임)
        vp_x, vp_y = self._detect_vp_from_video(cap, total_frames)
        if vp_x is None:
            cap.release()
            return {'error': 'VP detection failed'}

        # 관벽 ROI 마스크 생성 (VP에서 반경 60~90% 링 영역)
        roi_mask = self._create_pipe_wall_roi(img_w, img_h, vp_x, vp_y)

        # Motion scan: 매 scan_every_n 프레임마다 SSIM 계산
        motion_profile, stops = self._detect_stops(
            cap, total_frames, roi_mask,
            scan_every_n=scan_every_n,
            ssim_threshold=ssim_threshold,
            min_stop_frames=min_stop_frames,
            fps=fps,
            progress_callback=lambda cur, total:
                progress_callback(int(cur / max(total, 1) * 35), 100, f"스캔 {cur}/{total}")
                if progress_callback else None
        )

        cap.release()

        if not stops:
            return {
                'error': 'No stops detected',
                'video_path': str(video_path),
                'motion_profile': motion_profile,
                'stops': [],
            }

        # ═══ Phase 2: 정지 구간별 분석 ═══
        if progress_callback:
            progress_callback(35, 100, f"Phase 2: {len(stops)}개 정지 구간 분석 중...")

        # PipeUnwrapper 생성
        unwrap_h = 200
        output_width = 720
        strip_height = 12
        unwrapper = PipeUnwrapper(
            vp_x, vp_y, img_w, img_h,
            pipe_diameter_mm=pipe_diameter_mm,
            output_width=output_width,
            output_height=unwrap_h
        )

        y_start = int(unwrap_h * 0.15)
        y_end = int(unwrap_h * 0.65)

        cap = cv2.VideoCapture(video_path)
        total_stops = len(stops)

        frame_results = []
        strips_clean = []
        strips_overlay = []
        strips_annular = []
        strip_stops = []      # strip과 1:1 대응하는 stop (프레임 read 실패 시 정렬 유지용)
        vis_strips = []       # Global Area Ratio용 가시영역 이진 마스크 (band×θ)
        def_strips = []       # Global Area Ratio용 결함 이진 마스크 (band×θ)
        last_distance = None
        mask_ppm = None       # PPNet 전개 마스크의 pixel_per_mm (metric 스케일)

        for si, stop in enumerate(stops):
            # 가장 선명한 프레임 선택
            best_frame = self._select_best_frame(cap, stop, roi_mask)
            stop['best_frame'] = best_frame

            cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
            ret, frame = cap.read()
            if not ret:
                continue
            if self.preprocess_fn is not None:
                frame = self.preprocess_fn(frame)   # 렌즈 왜곡 보정 등

            # OSD 거리 (주입 시) — 직전 유효값 대비 100m 이상 점프는 OCR 오독으로 버림
            if self.distance_fn is not None:
                try:
                    d = self.distance_fn(frame)
                except Exception:
                    d = None
                if d is not None and last_distance is not None and abs(d - last_distance) >= 100:
                    d = None
                if d is not None:
                    last_distance = d
                stop['distance_m'] = d

            # YOLO 추론
            detections = self._infer_frame(frame)
            stop['detections'] = detections

            # 결함 집계
            by_class, total_ratio, num_objects = self._aggregate_detections(
                detections, img_w * img_h
            )
            stop['defects'] = by_class
            stop['total_defect_ratio'] = total_ratio
            stop['num_objects'] = num_objects

            # 프레임 결과 (비디오 오버레이용)
            frame_results.append({
                'frame_number': best_frame,
                'timestamp_sec': round(best_frame / fps, 2),
                'stop_index': si,
                'detections': detections,
            })

            if self.unwrap_fn is not None:
                # ══ PPNet 기반 전개 (GNU Mapping) ══
                uw = None
                try:
                    uw = self.unwrap_fn(frame, detections, pipe_diameter_mm)
                except Exception as e:
                    print(f"[Survey] unwrap_fn error @stop {si}: {e}")
                if uw is None:
                    continue
                vis_m = uw['visible_mask'].astype(np.uint8)
                def_m = uw['defect_mask'].astype(np.uint8) & vis_m
                mask_ppm = uw.get('pixel_per_mm', mask_ppm)
                vis_strips.append(vis_m)
                def_strips.append(def_m)
                strip_stops.append(stop)

                # 파노라마 호환: 전개 전체를 thin-band(가로 output_width)로 축약
                ub = uw['unwrapped_bgr']
                ov = uw.get('overlay_bgr')
                strips_clean.append(cv2.resize(ub, (output_width, strip_height)))
                strips_overlay.append(cv2.resize(ov if ov is not None else ub,
                                                 (output_width, strip_height)))
            else:
                # ══ 극좌표 전개 (PipeUnwrapper) — unwrap_fn 미주입 시 ══
                unwrapped_raw = unwrapper.unwrap(frame)
                unwrapped = self._color_correct(unwrapped_raw)

                crop = unwrapped[y_start:y_end, :, :]
                strip_clean = cv2.resize(crop, (output_width, strip_height))

                strip_over = strip_clean.copy()
                for det in detections:
                    polygon = det.get('polygon', [])
                    if polygon and len(polygon) >= 3:
                        self._draw_defect_on_strip(
                            strip_over, polygon, det.get('label', 'unknown'),
                            unwrapper, y_start, y_end, strip_height, output_width
                        )

                strips_clean.append(strip_clean)
                strips_overlay.append(strip_over)
                strip_stops.append(stop)

                # ── Global Area Ratio용 이진 마스크 (전개 band의 θ×반경) ──
                vis_full = (unwrapped_raw.sum(axis=2) > 12).astype(np.uint8)
                def_full = np.zeros((unwrap_h, output_width), dtype=np.uint8)
                for det in detections:
                    polygon = det.get('polygon', [])
                    if not polygon or len(polygon) < 3:
                        continue
                    if isinstance(polygon[0], (list, tuple)):
                        flat = []
                        for pt in polygon:
                            flat.extend([float(pt[0]), float(pt[1])])
                    else:
                        flat = [float(v) for v in polygon]
                    transformed = unwrapper.transform_polygon(flat)
                    pts = []
                    for j in range(0, len(transformed), 2):
                        tx = max(0, min(output_width - 1, transformed[j]))
                        ty = max(0, min(unwrap_h - 1, transformed[j + 1]))
                        pts.append([int(tx), int(ty)])
                    if len(pts) >= 3:
                        cv2.fillPoly(def_full, [np.array(pts, dtype=np.int32)], 1)
                def_full &= vis_full   # 가시영역 밖 결함은 제외
                vis_strips.append(vis_full[y_start:y_end, :].copy())
                def_strips.append(def_full[y_start:y_end, :].copy())

            # Annular ring strip
            ann_strip = self._extract_annular_strip(frame, vp_x, vp_y, 3)
            ann_strip = self._color_correct(ann_strip)
            strips_annular.append(ann_strip)

            if progress_callback:
                pct = 35 + int((si + 1) / total_stops * 50)
                progress_callback(pct, 100, f"Stop {si+1}/{total_stops} 분석")

        cap.release()

        # ═══ Phase 3: 후처리 ═══
        if progress_callback:
            progress_callback(88, 100, "Phase 3: 결과 저장 중...")

        # 이미지 저장
        output_dir = os.path.dirname(video_path)
        video_name = Path(video_path).stem

        panorama_path = None
        panorama_overlay_path = None
        stripmap_path = None

        if strips_clean:
            panorama = self._stack_strips_by_distance(strips_clean, strip_stops)
            panorama_over = self._stack_strips_by_distance(strips_overlay, strip_stops)
            panorama_path = os.path.join(output_dir, f'{video_name}_unwrap.jpg')
            panorama_overlay_path = os.path.join(output_dir, f'{video_name}_unwrap_overlay.jpg')
            cv2.imwrite(panorama_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(panorama_overlay_path, panorama_over, [cv2.IMWRITE_JPEG_QUALITY, 92])

        if strips_annular:
            strip_map = np.vstack(strips_annular)
            stripmap_path = os.path.join(output_dir, f'{video_name}_stripmap.jpg')
            cv2.imwrite(stripmap_path, strip_map, [cv2.IMWRITE_JPEG_QUALITY, 92])

        # ═══ Global Area Ratio — 구간 전체 중복 제거 통합 면적비 ═══
        global_area_ratio = self._compute_global_area_ratio(
            vis_strips, def_strips, strip_stops,
            pipe_diameter_mm=pipe_diameter_mm,
            strip_axial_mm=strip_axial_mm,
            px_per_mm=global_px_per_mm,
            mask_pixel_per_mm=mask_ppm,
            output_dir=output_dir, video_name=video_name,
        )

        coord_sys = unwrapper.get_coordinate_system()

        # 결함 통계
        defect_stops = [s for s in stops if s.get('total_defect_ratio', 0) > 0]
        by_class_summary = {}
        for stop in stops:
            for cls_name, cls_info in stop.get('defects', {}).items():
                if cls_name not in by_class_summary:
                    by_class_summary[cls_name] = {'stop_count': 0, 'max_ratio': 0.0}
                if cls_info.get('pixel_ratio', 0) > 0:
                    by_class_summary[cls_name]['stop_count'] += 1
                    by_class_summary[cls_name]['max_ratio'] = max(
                        by_class_summary[cls_name]['max_ratio'], cls_info['pixel_ratio'])

        total_stop_time = sum(s.get('duration_sec', 0) for s in stops)
        total_video_time = total_frames / fps if fps > 0 else 0

        summary = {
            'total_frames': total_frames,
            'fps': fps,
            'video_duration_sec': round(total_video_time, 1),
            'pipe_diameter_mm': pipe_diameter_mm,
            'ssim_threshold': ssim_threshold,
            'total_stops': len(stops),
            'defect_stops': len(defect_stops),
            'total_stop_time_sec': round(total_stop_time, 1),
            'stop_ratio_pct': round(total_stop_time / total_video_time * 100, 1) if total_video_time > 0 else 0,
            'analyzed_frames': len(frame_results),
            'by_class': by_class_summary,
            'global_area_ratio_pct': (global_area_ratio or {}).get('global_area_ratio_pct'),
        }

        if progress_callback:
            progress_callback(100, 100, "완료")

        return {
            'video_path': str(video_path),
            'pipe_diameter_mm': pipe_diameter_mm,
            'fps': fps,
            'total_frames': total_frames,
            'stops': stops,
            'motion_profile': motion_profile,
            'frame_results': frame_results,
            'summary': summary,
            'panorama_path': panorama_path,
            'panorama_overlay_path': panorama_overlay_path,
            'stripmap_path': stripmap_path,
            'global_area_ratio': global_area_ratio,
            'coordinate_system': coord_sys,
            'vp': {'x': vp_x, 'y': vp_y},
        }

    # ════════════════════════════════════════════
    #  Motion Detection
    # ════════════════════════════════════════════
    def _create_pipe_wall_roi(self, img_w, img_h, vp_x, vp_y):
        """관벽 ROI 마스크 생성 — VP에서 반경 60~90% 링 영역
        부유물(VP 근처)을 제외하고 관벽 텍스처만 포함
        """
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        # VP에서 프레임 가장자리까지 최소 거리
        max_r = min(vp_x, img_w - vp_x, vp_y, img_h - vp_y)

        # ROI 붕괴 방지: VP가 다소 치우쳐도 ring이 비지 않도록 반경 하한 보장
        # (off-center VP면 ring이 프레임 밖으로 일부 나가지만 cv2.circle이 자동 clip)
        min_r = int(min(img_w, img_h) * 0.30)
        if max_r < min_r:
            max_r = min_r

        r_inner = int(max_r * 0.60)
        r_outer = int(max_r * 0.90)

        cv2.circle(mask, (vp_x, vp_y), r_outer, 255, -1)
        cv2.circle(mask, (vp_x, vp_y), r_inner, 0, -1)

        # OSD 영역 제외 (상단 15%, 하단 15%)
        mask[:int(img_h * 0.15), :] = 0
        mask[int(img_h * 0.85):, :] = 0

        return mask

    def _detect_stops(self, cap, total_frames, roi_mask,
                      scan_every_n=25, ssim_threshold=0.92,
                      min_stop_frames=15, fps=25.0,
                      progress_callback=None):
        """SSIM 기반 정지/이동 구간 탐지 (최적화: 순차읽기 + 다운스케일)

        Returns:
            motion_profile: [{frame, ssim_score}, ...]
            stops: [{index, start_frame, end_frame, duration_sec, ...}, ...]
        """
        motion_profile = []

        # ROI bounding box
        roi_ys, roi_xs = np.where(roi_mask > 0)
        if len(roi_ys) == 0:
            return [], []

        ry1, ry2 = int(roi_ys.min()), int(roi_ys.max())
        rx1, rx2 = int(roi_xs.min()), int(roi_xs.max())

        # 다운스케일된 ROI 마스크 (1/4 크기) — SSIM win_size(7) 이상 보장
        scale = 0.25
        roi_crop = roi_mask[ry1:ry2+1, rx1:rx2+1]
        small_h = max(7, int(roi_crop.shape[0] * scale))
        small_w = max(7, int(roi_crop.shape[1] * scale))
        roi_crop_small = cv2.resize(roi_crop, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        # 작은 ROI(작은 영상)에서도 안전한 홀수 win_size
        ssim_win = min(7, small_h, small_w)
        if ssim_win % 2 == 0:
            ssim_win -= 1
        ssim_win = max(3, ssim_win)

        prev_small = None
        frame_count = 0
        total_scans = total_frames // max(scan_every_n, 1)

        # 순차 읽기 — grab()으로 빠르게 skip, 필요한 프레임만 retrieve()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while frame_count < total_frames:
            if frame_count % scan_every_n == 0:
                ret, frame = cap.read()
                if not ret:
                    break

                # ROI crop + 다운스케일 + grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_patch = gray[ry1:ry2+1, rx1:rx2+1]
                small = cv2.resize(roi_patch, (small_w, small_h), interpolation=cv2.INTER_AREA)
                small[roi_crop_small == 0] = 0

                if prev_small is not None:
                    score = ssim(prev_small, small, win_size=ssim_win, data_range=255)
                    motion_profile.append({
                        'frame': frame_count,
                        'ssim_score': round(float(score), 4),
                    })

                prev_small = small

                scan_idx = frame_count // scan_every_n
                if progress_callback and scan_idx % 20 == 0:
                    progress_callback(scan_idx, total_scans)
            else:
                # 디코딩 없이 건너뛰기 (훨씬 빠름)
                cap.grab()

            frame_count += 1

        # 정지 구간 추출: ssim >= threshold인 연속 구간
        stops = []
        in_stop = False
        stop_start = 0

        for mp in motion_profile:
            is_still = mp['ssim_score'] >= ssim_threshold

            if is_still and not in_stop:
                stop_start = mp['frame']
                in_stop = True
            elif not is_still and in_stop:
                stop_end = mp['frame']
                duration_frames = stop_end - stop_start
                if duration_frames >= min_stop_frames:
                    stops.append({
                        'index': len(stops),
                        'start_frame': stop_start,
                        'end_frame': stop_end,
                        'duration_frames': duration_frames,
                        'duration_sec': round(duration_frames / fps, 2),
                        'timestamp_sec': round(stop_start / fps, 2),
                    })
                in_stop = False

        # 마지막 구간 처리
        if in_stop and motion_profile:
            stop_end = motion_profile[-1]['frame']
            duration_frames = stop_end - stop_start
            if duration_frames >= min_stop_frames:
                stops.append({
                    'index': len(stops),
                    'start_frame': stop_start,
                    'end_frame': stop_end,
                    'duration_frames': duration_frames,
                    'duration_sec': round(duration_frames / fps, 2),
                    'timestamp_sec': round(stop_start / fps, 2),
                })

        return motion_profile, stops

    def _select_best_frame(self, cap, stop, roi_mask):
        """정지 구간 내에서 가장 선명한 프레임 선택 (Laplacian variance)"""
        start = stop['start_frame']
        end = stop['end_frame']
        duration = end - start

        # 구간 내 5개 지점 샘플링
        sample_count = min(5, max(1, duration // 10))
        sample_frames = np.linspace(start, end, sample_count, dtype=int)

        best_frame = (start + end) // 2
        best_sharpness = -1

        for fn in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fn))
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ROI 내 Laplacian variance = 선명도 지표
            masked = cv2.bitwise_and(gray, gray, mask=roi_mask)
            laplacian = cv2.Laplacian(masked, cv2.CV_64F)
            sharpness = laplacian.var()

            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame = int(fn)

        return best_frame

    # ════════════════════════════════════════════
    #  YOLO 추론
    # ════════════════════════════════════════════
    def _infer_frame(self, frame) -> list:
        """YOLO instance segmentation — infer_fn 주입 시 로컬, 아니면 GPU 서버 호출"""
        if self.infer_fn is not None:
            try:
                return self.infer_fn(frame)
            except Exception as e:
                print(f"[Survey] local inference error: {e}")
                return []
        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            resp = requests.post(
                f'{self.gpu_server_url}/api/survey/infer',
                json={'image_base64': img_b64},
                timeout=30
            )

            if resp.status_code != 200:
                return []

            data = resp.json()
            if not data.get('success'):
                return []

            return data.get('detections', [])

        except Exception as e:
            print(f"[Survey] YOLO inference error: {e}")
            return []

    def _aggregate_detections(self, detections, total_pixels):
        """detections → by_class, total_ratio, num_objects"""
        by_class = {}
        total_area = 0

        for det in detections:
            cls_name = det.get('label', 'unknown')
            area = det.get('area', 0)
            if cls_name not in by_class:
                by_class[cls_name] = {'count': 0, 'total_area': 0, 'pixel_ratio': 0.0}
            by_class[cls_name]['count'] += 1
            by_class[cls_name]['total_area'] += area
            total_area += area

        if total_pixels > 0:
            for cls_name in by_class:
                by_class[cls_name]['pixel_ratio'] = round(
                    by_class[cls_name]['total_area'] / total_pixels * 100, 2)

        total_ratio = round(total_area / total_pixels * 100, 2) if total_pixels > 0 else 0
        return by_class, total_ratio, len(detections)

    # ════════════════════════════════════════════
    #  VP 탐지
    # ════════════════════════════════════════════
    def _detect_vp_from_video(self, cap, total_frames):
        """영상 초반 프레임으로 VP 탐지, 중앙값 반환"""
        vp_frames = []
        # 초반 5개 위치에서 샘플
        sample_positions = np.linspace(0, min(total_frames - 1, 500), 5, dtype=int)
        for fn in sample_positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fn))
            ret, frame = cap.read()
            if ret:
                vp_frames.append(self._detect_vp_simple(frame))

        if not vp_frames:
            return None, None

        vp_x = int(np.median([v[0] for v in vp_frames]))
        vp_y = int(np.median([v[1] for v in vp_frames]))

        # 퇴화 VP 방어: 가장자리(프레임 10% 이내)에 잡히면 탐지 실패로 보고 중심으로 폴백.
        # (관 CCTV는 소실점이 대략 화면 안쪽 → 가장자리 VP는 휴리스틱 오작동)
        cap_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH); cap_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if cap_w > 0 and cap_h > 0:
            mx, my = int(cap_w * 0.10), int(cap_h * 0.10)
            if not (mx <= vp_x <= cap_w - mx and my <= vp_y <= cap_h - my):
                print(f"[Survey] VP({vp_x},{vp_y}) 가장자리 → 중심 폴백")
                vp_x, vp_y = int(cap_w / 2), int(cap_h / 2)
        return vp_x, vp_y

    def _detect_vp_simple(self, frame):
        """간단한 VP(소실점) 탐지 — 화면 내부의 가장 어두운 영역 중심.
        상하좌우 가장자리 15%를 제외하여 비네팅/모서리 암부에 VP가 끌려가는 것을 방지."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray[:int(h * 0.15), :] = 255
        gray[int(h * 0.85):, :] = 255
        gray[:, :int(w * 0.15)] = 255
        gray[:, int(w * 0.85):] = 255
        blurred = cv2.GaussianBlur(gray, (w // 4 * 2 + 1, h // 4 * 2 + 1), 0)
        _, _, min_loc, _ = cv2.minMaxLoc(blurred)
        return min_loc

    # ════════════════════════════════════════════
    #  전개도 조립 (거리 비례 간격)
    # ════════════════════════════════════════════
    def _stack_strips_by_distance(self, strips, strip_stops,
                                  gap_avg_px=36, label_w=64, bg=(24, 24, 24)):
        """strip들을 거리(없으면 시각) 간격에 비례한 세로 간격으로 쌓는다.

        - 축 우선순위: OSD 거리(distance_m, 단위 m) > 영상 시각(timestamp_sec, 단위 s)
        - 층(strip) 높이는 그대로 — 간격(빈 공간)이 축값 차에 비례
        - 왼쪽 label_w 여백에 축값 라벨(거리 m / 시각 s / 둘 다 없으면 #index)
        - 유효 축값 2개 미만이면 간격 없이 쌓음
        """
        n = len(strips)
        if n == 0:
            return None

        w = strips[0].shape[1]
        dists = [s.get('distance_m') for s in strip_stops]
        times = [s.get('timestamp_sec') for s in strip_stops]

        # 거리 우선, 없으면 시각으로 fallback
        if sum(d is not None for d in dists) >= 2:
            axis, fmt = dists, lambda v: f"{v:.2f}m"
        elif sum(t is not None for t in times) >= 2:
            axis, fmt = times, lambda v: f"{v:.0f}s"
        else:
            axis, fmt = [None] * n, None

        known = [(i, v) for i, v in enumerate(axis) if v is not None]
        if len(known) >= 2:
            xs = [i for i, _ in known]
            ys = [v for _, v in known]
            filled = list(np.interp(np.arange(n), xs, ys))
            gaps = [abs(filled[i + 1] - filled[i]) for i in range(n - 1)]
            total = sum(gaps)
            if total > 0:
                k = gap_avg_px * (n - 1) / total
                gaps_px = [max(2, int(round(g * k))) for g in gaps]
            else:
                gaps_px = [2] * (n - 1)
        else:
            gaps_px = [0] * max(0, n - 1)

        rows = []
        for i, strip in enumerate(strips):
            h = strip.shape[0]
            row = np.full((h, label_w + w, 3), bg, dtype=np.uint8)
            row[:, label_w:] = strip
            v = axis[i]
            label = fmt(v) if (v is not None and fmt) else f"#{strip_stops[i].get('index', i)}"
            cv2.putText(row, label, (2, h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (200, 200, 200), 1, cv2.LINE_AA)
            rows.append(row)
            if i < n - 1 and gaps_px[i] > 0:
                rows.append(np.full((gaps_px[i], label_w + w, 3), bg, dtype=np.uint8))

        return np.vstack(rows)

    # ════════════════════════════════════════════
    #  Global Area Ratio — 구간 전체 중복 제거 통합 면적비
    # ════════════════════════════════════════════
    def _compute_global_area_ratio(self, vis_strips, def_strips, strip_stops,
                                   pipe_diameter_mm=300, strip_axial_mm=None,
                                   px_per_mm=0.2, mask_pixel_per_mm=None,
                                   output_dir=None, video_name=None):
        """정지 구간 strip들을 거리(z)축에 중복 제거 누적 → 통합 면적비.

            Global Area Ratio = Σ(unique 결함 셀) / Σ(unique 가시 셀)   [θ×z 캔버스]

        같은 z 위치를 본 strip들은 같은 캔버스 행에 OR로 겹쳐져 중복이 제거된다.
        전개 캔버스는 θ(원주) 균등 × z(축) metric → 각 셀의 실면적이 균일하므로
        셀 개수 비 = 실면적 비.

        모드:
          - OSD 거리(distance_m) 2개 이상 → metric 모드 (mm 단위 거리축 중복 제거)
          - 거리 없음 → aggregate 모드: 인위적 중복 없이 구간 가시면적 가중 평균
            (정지 반복 중복은 대표 프레임 1장으로 이미 제거, 공간 중복은 거리 없이 추정 불가)

        mask_pixel_per_mm: PPNet(GNU Mapping) 전개 마스크의 px/mm. 지정되면
            마스크가 이미 metric θ×z(strip 높이 = max_depth_mm·ppm)이므로 resize 없이
            그 스케일로 누적하고 strip_axial_mm을 마스크 높이에서 역산(=max_depth_mm).
        strip_axial_mm: (극좌표 전개용) 정지 1장이 대표하는 관 축방향 길이(mm). 기본=관 직경.
        """
        n = len(vis_strips)
        if n == 0:
            return None

        band_h, w = vis_strips[0].shape

        # 캔버스 px/mm 결정. PPNet 마스크는 자체 metric 스케일을 그대로 사용.
        ppnet_mode = mask_pixel_per_mm is not None and mask_pixel_per_mm > 0
        if ppnet_mode:
            canvas_ppm = float(mask_pixel_per_mm)
            strip_axial_mm = band_h / canvas_ppm          # = max_depth_mm
            strip_full_px = band_h                          # PPNet strip 높이 그대로
        else:
            canvas_ppm = px_per_mm
            if strip_axial_mm is None or strip_axial_mm <= 0:
                strip_axial_mm = float(pipe_diameter_mm)
            strip_full_px = max(1, int(round(strip_axial_mm * canvas_ppm)))

        # ── 거리(z)축 결정: OSD 거리 2개 이상이면 metric, 아니면 aggregate ──
        dists = [s.get('distance_m') for s in strip_stops]
        n_dist = sum(d is not None for d in dists)

        metric = False
        axis = None
        if n_dist >= 2:
            metric = True
            axis = [(d * 1000.0 if d is not None else None) for d in dists]  # m→mm
            # 결측 거리는 알려진 값으로 선형 보간
            known = [(i, v) for i, v in enumerate(axis) if v is not None]
            xs = [i for i, _ in known]
            ys = [float(v) for _, v in known]
            axis = list(np.interp(np.arange(n), xs, ys))

        # ── strip 배치 (top row) ──
        if metric and axis is not None:
            strip_axial_px = strip_full_px
            z0 = min(axis)
            tops = [int(round((z - z0) * canvas_ppm)) for z in axis]
            confidence = 'metric'
            z_span_mm = float(max(axis) - z0)
        else:
            # 거리 정보 없음 → 인위적 공간 중복 없이 구간 가시면적 가중 평균.
            # (같은 자리 정지 반복 중복은 이미 대표 프레임 1장으로 제거된 상태이고,
            #  인접 stop 간 공간 중복은 거리 없이는 알 수 없으므로 중복 제거를 적용하지 않는다.
            #  timestamp 비례 배치는 정지 중에도 시각이 흘러 인위적 중복을 만들므로 사용하지 않음)
            strip_axial_px = strip_full_px
            tops = [i * strip_axial_px for i in range(n)]
            confidence = 'aggregate(no-distance)'
            z_span_mm = None

        canvas_rows = max(t + strip_axial_px for t in tops)
        # 과도한 캔버스 방지 (>24000행이면 자동 축소)
        if canvas_rows > 24000:
            shrink = 24000.0 / canvas_rows
            tops = [int(round(t * shrink)) for t in tops]
            strip_axial_px = max(1, int(round(strip_axial_px * shrink)))
            canvas_ppm = canvas_ppm * shrink
            canvas_rows = max(t + strip_axial_px for t in tops)

        canvas_vis = np.zeros((canvas_rows, w), dtype=np.uint8)
        canvas_def = np.zeros((canvas_rows, w), dtype=np.uint8)

        raw_vis = 0   # 중복 제거 전 가시 셀 합 (overlap factor 계산용)
        raw_def = 0
        for i in range(n):
            vs = cv2.resize(vis_strips[i], (w, strip_axial_px), interpolation=cv2.INTER_NEAREST)
            ds = cv2.resize(def_strips[i], (w, strip_axial_px), interpolation=cv2.INTER_NEAREST)
            t = tops[i]
            sl = slice(t, t + strip_axial_px)
            canvas_vis[sl] |= vs
            canvas_def[sl] |= ds
            raw_vis += int(vs.sum())
            raw_def += int(ds.sum())

        # 결함은 가시영역 안에서만 인정
        canvas_def &= canvas_vis
        visible = int(canvas_vis.sum())
        defect = int(canvas_def.sum())
        ratio = round(defect / visible * 100, 3) if visible > 0 else 0.0

        # ── 시각화 저장 (가시=회색, 결함=빨강) ──
        canvas_path = None
        if output_dir and video_name:
            viz = np.zeros((canvas_rows, w, 3), dtype=np.uint8)
            viz[canvas_vis > 0] = (70, 70, 70)
            viz[canvas_def > 0] = (40, 40, 220)
            canvas_path = os.path.join(output_dir, f'{video_name}_global_area.jpg')
            cv2.imwrite(canvas_path, viz, [cv2.IMWRITE_JPEG_QUALITY, 88])

        return {
            'global_area_ratio_pct': ratio,
            'visible_cells': visible,
            'defect_cells': defect,
            'naive_sum_ratio_pct': round(raw_def / raw_vis * 100, 3) if raw_vis > 0 else 0.0,
            'overlap_factor': round(raw_vis / visible, 2) if visible > 0 else None,
            'metric': metric,
            'confidence': confidence,
            'unwrap': 'ppnet' if ppnet_mode else 'polar',
            'strip_axial_mm': round(float(strip_axial_mm), 1),
            'px_per_mm': round(float(canvas_ppm), 4),
            'z_span_mm': round(z_span_mm, 1) if z_span_mm is not None else None,
            'num_stops': n,
            'canvas_rows': int(canvas_rows),
            'canvas_path': canvas_path,
        }

    # ════════════════════════════════════════════
    #  전개도 오버레이
    # ════════════════════════════════════════════
    def _draw_defect_on_strip(self, strip, polygon, label, unwrapper,
                              y_start, y_end, strip_height, output_width):
        """전개 strip 위에 결함 폴리곤 오버레이"""
        color = _get_defect_color(label, self.colors)

        if isinstance(polygon[0], (list, tuple)):
            flat = []
            for pt in polygon:
                flat.extend([float(pt[0]), float(pt[1])])
        else:
            flat = [float(v) for v in polygon]

        transformed = unwrapper.transform_polygon(flat)

        strip_pts = []
        for j in range(0, len(transformed), 2):
            tx = transformed[j]
            ty = transformed[j + 1]
            sy = (ty - y_start) / (y_end - y_start) * strip_height
            sy = max(0, min(strip_height - 1, sy))
            sx = max(0, min(output_width - 1, tx))
            strip_pts.append([int(sx), int(sy)])

        if len(strip_pts) >= 3:
            pts_np = np.array(strip_pts, dtype=np.int32)
            overlay_layer = strip.copy()
            cv2.fillPoly(overlay_layer, [pts_np], color)
            cv2.addWeighted(overlay_layer, 0.35, strip, 0.65, 0, strip)
            cv2.polylines(strip, [pts_np], True, color, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════
    #  Annular ring strip
    # ════════════════════════════════════════════
    def _extract_annular_strip(self, frame, vp_x, vp_y, strip_height=1):
        h, w = frame.shape[:2]

        max_radius = int(min(vp_x, w - vp_x, vp_y, h - vp_y) * 0.95)
        if max_radius < 50:
            max_radius = min(h, w) // 2

        output_width = 720
        polar = cv2.warpPolar(
            frame, (max_radius, output_width),
            (vp_x, vp_y), max_radius, cv2.WARP_POLAR_LINEAR
        )

        r_inner = int(max_radius * 0.75)
        r_outer = int(max_radius * 0.95)

        ring = polar[:, r_inner:r_outer, :]
        ring_t = ring.transpose(1, 0, 2)
        strip = cv2.resize(ring_t, (ring_t.shape[1], strip_height))
        return strip

    # ════════════════════════════════════════════
    #  색보정
    # ════════════════════════════════════════════
    def _color_correct(self, img):
        """그레이월드 WB + CLAHE"""
        b, g, r = cv2.split(img)
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        avg_all = (avg_b + avg_g + avg_r) / 3
        if avg_b > 0 and avg_g > 0 and avg_r > 0:
            b = np.clip(b * (avg_all / avg_b), 0, 255).astype(np.uint8)
            g = np.clip(g * (avg_all / avg_g), 0, 255).astype(np.uint8)
            r = np.clip(r * (avg_all / avg_r), 0, 255).astype(np.uint8)
        img = cv2.merge([b, g, r])
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 2))
        l_ch = clahe.apply(l_ch)
        img = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        return img


# ─── CLI ───

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipe_survey.py <video_path> [--ssim-threshold 0.92] [--min-stop-frames 15] [--scan-every 5]")
        sys.exit(1)

    args = sys.argv[1:]
    ssim_thresh = 0.92
    min_stop = 15
    scan_every = 5
    video_path = None

    i = 0
    while i < len(args):
        if args[i] == '--ssim-threshold':
            ssim_thresh = float(args[i + 1]); i += 2
        elif args[i] == '--min-stop-frames':
            min_stop = int(args[i + 1]); i += 2
        elif args[i] == '--scan-every':
            scan_every = int(args[i + 1]); i += 2
        else:
            video_path = args[i]; i += 1

    if not video_path:
        print("Error: video path required")
        sys.exit(1)

    analyzer = PipeSurveyAnalyzer(gpu=True)

    print(f"Analyzing: {os.path.basename(video_path)}")
    t0 = time.time()

    def progress(cur, total, msg):
        print(f"  [{msg}] {cur}%")

    result = analyzer.analyze_video(
        video_path,
        ssim_threshold=ssim_thresh,
        min_stop_frames=min_stop,
        scan_every_n=scan_every,
        progress_callback=progress
    )

    elapsed = time.time() - t0

    if 'error' in result:
        print(f"Error: {result['error']}")
        if result.get('stops') is not None:
            print(f"Stops found: {len(result.get('stops', []))}")
        sys.exit(1)

    s = result['summary']
    print(f"\n{'=' * 50}")
    print(f"관경: {s['pipe_diameter_mm']}mm")
    print(f"영상: {s['video_duration_sec']}s ({s['total_frames']} frames)")
    print(f"정지 구간: {s['total_stops']}개 ({s['total_stop_time_sec']}s, {s['stop_ratio_pct']}%)")
    print(f"결함 구간: {s['defect_stops']}개")
    print(f"소요: {elapsed:.1f}s")

    print(f"\nStop 목록:")
    for stop in result['stops'][:20]:
        ts = stop['timestamp_sec']
        dur = stop['duration_sec']
        ratio = stop.get('total_defect_ratio', 0)
        dets = stop.get('num_objects', 0)
        print(f"  #{stop['index']:3d} | {ts:7.1f}s | {dur:5.1f}s | defect {ratio:5.1f}% | {dets} objects")

    out_path = video_path + '.survey.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n저장: {out_path}")
