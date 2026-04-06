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
from pathlib import Path

from osd_ocr import OSDDistanceReader


class PipeSurveyAnalyzer:
    """영상 전체를 분석하여 구간별 결함 분포를 생성한다."""

    def __init__(self, gpu=True, segment_model=None):
        """
        Args:
            gpu: GPU 사용 여부
            segment_model: 세그멘테이션 모델 (None이면 OSD+거리만)
        """
        self.osd_reader = OSDDistanceReader(gpu=gpu)
        self.segment_model = segment_model

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

        # Phase 3: 결함 분석 (모델이 있을 때)
        # TODO: 세그멘테이션 모델로 각 구간 프레임 분석
        # 현재는 구조만 잡아둠

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
        }

        cap.release()

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

    def generate_strip_map_frames(self, video_path, distances,
                                  strip_height_px=5, output_dir=None) -> str:
        """Strip-map 생성: 각 프레임에서 중앙 수평 띠를 추출하여 이어붙임

        Args:
            video_path: 영상 경로
            distances: frame_distances 리스트
            strip_height_px: 추출할 띠 높이 (px)
            output_dir: 출력 디렉토리

        Returns:
            저장된 strip-map 이미지 경로
        """
        if not distances:
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 거리 변화가 있는 프레임만 추출 (중복 제거)
        unique_distances = []
        last_d = None
        for d in distances:
            if last_d is None or abs(d['distance_m'] - last_d) > 0.01:
                unique_distances.append(d)
                last_d = d['distance_m']

        strips = []
        # 중앙 수평 띠 위치 (프레임 높이의 50% 부근)
        strip_y = frame_h // 2 - strip_height_px // 2

        for d in unique_distances:
            cap.set(cv2.CAP_PROP_POS_FRAMES, d['frame_number'])
            ret, frame = cap.read()
            if not ret:
                continue

            # 중앙 수평 띠 추출
            strip = frame[strip_y:strip_y + strip_height_px, :, :]
            strips.append(strip)

        cap.release()

        if not strips:
            return None

        # 세로로 이어붙임
        strip_map = np.vstack(strips)

        # 저장
        if output_dir is None:
            output_dir = os.path.dirname(video_path)

        video_name = Path(video_path).stem
        out_path = os.path.join(output_dir, f'{video_name}_stripmap.jpg')
        cv2.imwrite(out_path, strip_map, [cv2.IMWRITE_JPEG_QUALITY, 90])

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
