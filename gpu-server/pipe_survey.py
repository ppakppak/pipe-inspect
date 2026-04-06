#!/usr/bin/env python3
"""
관로 조사 통합 모듈 (Pipe Survey Analyzer)

OSD 거리 + AI 세그멘테이션 + 면적비 산출을 결합하여
전체 관로의 구간별 결함 분포를 생성한다.

출력:
  - 구간별 결함 통계 (1m 단위)
  - 결함 분포 JSON
  - Strip-map용 프레임-거리 매핑
"""

import cv2
import numpy as np
import json
import os
import time
import requests
import base64
from pathlib import Path

from osd_ocr import OSDDistanceReader


class PipeSurveyAnalyzer:
    """영상 전체를 분석하여 구간별 결함 분포를 생성한다."""

    def __init__(self, gpu=True, gpu_server_url='http://localhost:5004'):
        """
        Args:
            gpu: GPU 사용 여부
            gpu_server_url: GPU 서버 API URL
        """
        self.osd_reader = OSDDistanceReader(gpu=gpu)
        self.gpu_server_url = gpu_server_url

    def analyze_video(self, video_path, pipe_diameter_mm=300,
                      sample_interval=15, section_length_m=1.0,
                      progress_callback=None) -> dict:
        """영상 전체 분석

        Args:
            video_path: 영상 파일 경로
            pipe_diameter_mm: 관경 (mm)
            sample_interval: OCR 수행 간격 (프레임 수)
            section_length_m: 통계 구간 길이 (m)
            progress_callback: fn(frame_num, total_frames, message)

        Returns:
            {
                'video_path': str,
                'pipe_diameter_mm': int,
                'total_distance_m': float,
                'total_frames': int,
                'frame_distances': [{frame_number, distance_m, timestamp_sec}, ...],
                'sections': [
                    {
                        'start_m': float, 'end_m': float,
                        'frame_range': [start_frame, end_frame],
                        'defects': {class_name: {'count': int, 'area_ratio': float}},
                        'total_defect_ratio': float,
                    }, ...
                ],
                'summary': {
                    'total_length_m': float,
                    'defect_length_m': float,
                    'defect_ratio': float,
                    'by_class': {class_name: {'length_m': float, 'max_ratio': float}},
                }
            }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Phase 1: OSD 거리 스캔
        if progress_callback:
            progress_callback(0, total_frames, "Phase 1: OSD 거리 스캔 중...")

        distances = self.osd_reader.read_video_distances(
            video_path, sample_interval,
            lambda fn, total: progress_callback(fn, total, "OSD 스캔") if progress_callback else None
        )

        if not distances:
            cap.release()
            return {'error': 'No distance readings found', 'frame_distances': []}

        # 거리 보간 — 모든 프레임에 거리 매핑
        frame_dist_map = self._interpolate_distances(distances, total_frames)

        # Phase 2: 구간 분할
        total_dist = distances[-1]['distance_m'] - distances[0]['distance_m']
        start_dist = distances[0]['distance_m']
        end_dist = distances[-1]['distance_m']
        num_sections = max(1, int(np.ceil(total_dist / section_length_m)))

        sections = []
        for i in range(num_sections):
            sec_start = start_dist + i * section_length_m
            sec_end = min(sec_start + section_length_m, end_dist)

            # 해당 구간의 프레임 범위
            frames_in_section = [
                fn for fn, d in frame_dist_map.items()
                if sec_start <= d < sec_end
            ]

            section = {
                'index': i,
                'start_m': round(sec_start, 2),
                'end_m': round(sec_end, 2),
                'frame_range': [min(frames_in_section), max(frames_in_section)] if frames_in_section else [0, 0],
                'frame_count': len(frames_in_section),
                'defects': {},
                'total_defect_ratio': 0.0,
            }
            sections.append(section)

        # Phase 3: 결함 분석 — 구간별 대표 프레임 세그멘테이션
        if progress_callback:
            progress_callback(0, len(sections), "Phase 2: 결함 분석 중...")

        cap = cv2.VideoCapture(video_path)
        for si, section in enumerate(sections):
            if section['frame_count'] == 0:
                continue

            # 구간 중간 프레임을 대표로 추론
            mid_frame = (section['frame_range'][0] + section['frame_range'][1]) // 2
            defect_info = self._analyze_frame(cap, mid_frame)

            if defect_info:
                section['defects'] = defect_info.get('by_class', {})
                section['total_defect_ratio'] = defect_info.get('total_ratio', 0.0)
                section['num_objects'] = defect_info.get('num_objects', 0)

            if progress_callback:
                progress_callback(si + 1, len(sections), "결함 분석")

        cap.release()

        # 결함 통계
        defect_sections = [s for s in sections if s['total_defect_ratio'] > 0]
        defect_length = len(defect_sections) * section_length_m

        by_class_summary = {}
        for sec in sections:
            for cls_name, cls_info in sec.get('defects', {}).items():
                if cls_name not in by_class_summary:
                    by_class_summary[cls_name] = {'section_count': 0, 'max_ratio': 0.0}
                if cls_info.get('pixel_ratio', 0) > 0:
                    by_class_summary[cls_name]['section_count'] += 1
                    by_class_summary[cls_name]['max_ratio'] = max(
                        by_class_summary[cls_name]['max_ratio'], cls_info['pixel_ratio'])

        # 요약 생성
        summary = {
            'total_length_m': round(total_dist, 2),
            'start_m': round(start_dist, 2),
            'end_m': round(end_dist, 2),
            'total_frames': total_frames,
            'readings_count': len(distances),
            'sections_count': num_sections,
            'section_length_m': section_length_m,
            'pipe_diameter_mm': pipe_diameter_mm,
            'fps': fps,
            'defect_sections': len(defect_sections),
            'defect_length_m': round(defect_length, 2),
            'defect_ratio_pct': round(defect_length / total_dist * 100, 1) if total_dist > 0 else 0,
            'by_class': by_class_summary,
        }

        return {
            'video_path': str(video_path),
            'pipe_diameter_mm': pipe_diameter_mm,
            'frame_distances': distances,
            'sections': sections,
            'summary': summary,
        }

    def _interpolate_distances(self, distances, total_frames):
        """OSD 읽은 지점 사이를 선형 보간하여 전체 프레임-거리 매핑 생성"""
        frame_dist = {}

        for i, d in enumerate(distances):
            frame_dist[d['frame_number']] = d['distance_m']

        # 보간: 읽은 지점 사이를 선형으로 채움
        sorted_readings = sorted(distances, key=lambda d: d['frame_number'])

        for i in range(len(sorted_readings) - 1):
            f1 = sorted_readings[i]['frame_number']
            f2 = sorted_readings[i+1]['frame_number']
            d1 = sorted_readings[i]['distance_m']
            d2 = sorted_readings[i+1]['distance_m']

            if f2 > f1:
                for f in range(f1, f2 + 1):
                    ratio = (f - f1) / (f2 - f1)
                    frame_dist[f] = d1 + ratio * (d2 - d1)

        return frame_dist

    def _analyze_frame(self, cap, frame_number) -> dict:
        """단일 프레임 세그멘테이션 — GPU 서버 API 호출"""
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            return None

        try:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            img_b64 = base64.b64encode(buffer).decode('utf-8')

            resp = requests.post(
                f'{self.gpu_server_url}/api/ai/inference_raw',
                json={'image_base64': img_b64},
                timeout=30
            )

            if resp.status_code != 200:
                return None

            data = resp.json()
            if not data.get('success'):
                return None

            boxes = data.get('bounding_boxes', [])
            width = data.get('width', 1)
            height = data.get('height', 1)
            total_pixels = width * height

            by_class = {}
            total_defect_pixels = 0

            for box in boxes:
                cls_name = box.get('class_name', f"class_{box.get('class_id', '?')}")
                area = box.get('area', 0)
                if cls_name not in by_class:
                    by_class[cls_name] = {'count': 0, 'total_area': 0, 'pixel_ratio': 0.0}
                by_class[cls_name]['count'] += 1
                by_class[cls_name]['total_area'] += area
                total_defect_pixels += area

            for cls_name in by_class:
                by_class[cls_name]['pixel_ratio'] = round(
                    by_class[cls_name]['total_area'] / total_pixels * 100, 2)

            return {
                'by_class': by_class,
                'total_ratio': round(total_defect_pixels / total_pixels * 100, 2),
                'num_objects': len(boxes),
            }
        except Exception as e:
            return None

    def _detect_vp_simple(self, frame):
        """간단한 VP(소실점) 탐지 — 가장 어두운 영역의 중심"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # 상단 15% OSD 마스킹
        gray[:int(h*0.15), :] = 255
        # 하단 15% OSD 마스킹
        gray[int(h*0.85):, :] = 255
        # 대블러 후 최소밝기점
        blurred = cv2.GaussianBlur(gray, (w//4*2+1, h//4*2+1), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
        return min_loc  # (x, y)

    def _extract_annular_strip(self, frame, vp_x, vp_y, strip_height=1):
        """프레임에서 VP 중심 annular ring을 추출하여 펼친 띠를 반환

        Args:
            frame: BGR 프레임
            vp_x, vp_y: 소실점 좌표
            strip_height: 출력 띠 높이 (px)

        Returns:
            strip: (strip_height, output_width, 3) BGR 이미지
                   가로 = 관 둘레 (0~360도), 세로 = 반경 방향 두께
        """
        h, w = frame.shape[:2]

        # 최대 반경 = VP에서 프레임 가장자리까지 최소 거리
        max_radius = int(min(vp_x, w - vp_x, vp_y, h - vp_y) * 0.95)
        if max_radius < 50:
            max_radius = min(h, w) // 2

        # 극좌표 변환 (warpPolar)
        # 출력: rows = 각도(0~360), cols = 반경(0~max_radius)
        output_width = 720  # 360도를 720px로 (0.5도/px)
        polar = cv2.warpPolar(
            frame,
            (max_radius, output_width),  # (cols=radius, rows=angle)
            (vp_x, vp_y),
            max_radius,
            cv2.WARP_POLAR_LINEAR
        )
        # polar shape: (output_width, max_radius, 3)
        # 행 = 각도, 열 = 반경 (0=VP, max_radius=가장자리)

        # 카메라 바로 앞 관벽 = 바깥쪽 고리 (반경 70~85%)
        r_inner = int(max_radius * 0.55)
        r_outer = int(max_radius * 0.90)

        # 해당 반경 범위의 띠 추출
        ring = polar[:, r_inner:r_outer, :]  # (720, thickness, 3)

        # 반경 방향으로 평균 → 1px 높이 띠, 또는 리사이즈
        # ring shape: (720, thickness, 3) — 행=각도, 열=반경
        # 반경 방향을 strip_height로 리사이즈 (텍스처 보존)
        # 전치: (thickness, 720, 3) → resize → (strip_height, 720, 3)
        ring_t = ring.transpose(1, 0, 2)  # (thickness, 720, 3)
        strip = cv2.resize(ring_t, (ring_t.shape[1], strip_height))  # (strip_height, 720, 3)

        return strip

    def generate_strip_map_frames(self, video_path, distances,
                                  strip_height_px=3, output_dir=None,
                                  sample_every_n=1) -> str:
        """Annular Ring Strip-map 생성

        각 프레임에서 VP 중심의 바깥쪽 고리를 추출 → 펼쳐서 → 거리순으로 세로 이어붙임
        결과: 가로 = 관 둘레(0~360도), 세로 = 관로 거리(m)

        Args:
            video_path: 영상 경로
            distances: frame_distances 리스트
            strip_height_px: 각 프레임에서 추출할 띠 높이
            output_dir: 출력 디렉토리
            sample_every_n: N개 거리 포인트마다 1개 추출

        Returns:
            저장된 strip-map 이미지 경로
        """
        if not distances:
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        # VP 탐지 (처음 몇 프레임으로)
        vp_frames = []
        for d in distances[:5]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, d['frame_number'])
            ret, frame = cap.read()
            if ret:
                vp_frames.append(self._detect_vp_simple(frame))

        if not vp_frames:
            cap.release()
            return None

        # VP 중앙값
        vp_x = int(np.median([v[0] for v in vp_frames]))
        vp_y = int(np.median([v[1] for v in vp_frames]))

        # 프레임 보간: 거리 reading 사이를 선형 보간하여 촘촘하게 샘플링
        sorted_d = sorted(distances, key=lambda x: x['frame_number'])
        first_frame = sorted_d[0]['frame_number']
        last_frame = sorted_d[-1]['frame_number']
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 매 N프레임마다 strip 추출 (기본 50프레임 = 2초마다)
        frame_step = max(50, (last_frame - first_frame) // 500)  # 최대 500줄

        strips = []
        for fn in range(first_frame, last_frame + 1, frame_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue

            strip = self._extract_annular_strip(frame, vp_x, vp_y, strip_height_px)
            strips.append(strip)

        cap.release()

        if not strips:
            return None

        # 세로로 이어붙임 (세로 = 거리, 가로 = 둘레)
        strip_map = np.vstack(strips)

        # 저장
        if output_dir is None:
            output_dir = os.path.dirname(video_path)

        video_name = Path(video_path).stem
        out_path = os.path.join(output_dir, f'{video_name}_stripmap.jpg')
        cv2.imwrite(out_path, strip_map, [cv2.IMWRITE_JPEG_QUALITY, 92])

        return out_path


# ─── CLI ───

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipe_survey.py <video_path> [--interval N] [--section-length M] [--strip-map]")
        sys.exit(1)

    args = sys.argv[1:]
    interval = 30
    section_length = 1.0
    do_strip = False
    video_path = None

    i = 0
    while i < len(args):
        if args[i] == '--interval':
            interval = int(args[i+1]); i += 2
        elif args[i] == '--section-length':
            section_length = float(args[i+1]); i += 2
        elif args[i] == '--strip-map':
            do_strip = True; i += 1
        else:
            video_path = args[i]; i += 1

    if not video_path:
        print("Error: video path required")
        sys.exit(1)

    analyzer = PipeSurveyAnalyzer(gpu=True)

    print(f"Analyzing: {os.path.basename(video_path)}")
    t0 = time.time()

    def progress(fn, total, msg):
        if fn % (interval * 10) == 0:
            pct = fn / total * 100 if total > 0 else 0
            print(f"  [{msg}] {fn}/{total} ({pct:.0f}%)")

    result = analyzer.analyze_video(
        video_path,
        sample_interval=interval,
        section_length_m=section_length,
        progress_callback=progress
    )

    elapsed = time.time() - t0

    if 'error' in result:
        print(f"Error: {result['error']}")
        sys.exit(1)

    s = result['summary']
    print(f"\n{'='*50}")
    print(f"관로: {s['pipe_diameter_mm']}mm, {s['total_length_m']}m")
    print(f"구간: {s['start_m']}m → {s['end_m']}m ({s['sections_count']}구간 × {s['section_length_m']}m)")
    print(f"프레임: {s['total_frames']} (OCR {s['readings_count']}회)")
    print(f"소요: {elapsed:.1f}s")

    # 구간 미리보기
    print(f"\n구간별 현황 (상위 10):")
    for sec in result['sections'][:10]:
        print(f"  {sec['start_m']:6.1f}~{sec['end_m']:6.1f}m | frames {sec['frame_range'][0]}~{sec['frame_range'][1]} ({sec['frame_count']})")

    if len(result['sections']) > 10:
        print(f"  ... ({len(result['sections'])} 구간 총)")

    # JSON 저장
    out_path = video_path + '.survey.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out_path}")

    # Strip-map 생성
    if do_strip:
        print("\nStrip-map 생성 중...")
        strip_path = analyzer.generate_strip_map_frames(
            video_path, result['frame_distances']
        )
        if strip_path:
            print(f"Strip-map 저장: {strip_path}")
