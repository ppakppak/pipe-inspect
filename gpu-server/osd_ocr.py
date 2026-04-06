#!/usr/bin/env python3
"""
OSD 거리 OCR 모듈 (Pipe CCTV Distance Reader)

파이프 CCTV 영상 OSD에서 카메라 진행 거리(m)를 추출.
EasyOCR 기반으로 높은 인식률 보장.

OSD 포맷: 우하단 "NNN.NNM" (예: 000.54M), 좌상단 날짜/시간
"""

import cv2
import numpy as np
import re
import json
import os
import time
from pathlib import Path

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False


class OSDDistanceReader:
    """프레임 OSD에서 거리(m) 값을 읽는다."""

    DISTANCE_ROI = {'x_start': 0.70, 'x_end': 1.0, 'y_start': 0.85, 'y_end': 1.0}
    DATETIME_ROI = {'x_start': 0.0, 'x_end': 0.40, 'y_start': 0.0, 'y_end': 0.12}

    def __init__(self, gpu=True, debug=False):
        self.debug = debug
        self._reader = None
        self._gpu = gpu

    def _ensure_reader(self):
        if self._reader is None:
            if not HAS_EASYOCR:
                raise ImportError("easyocr not installed. pip install easyocr")
            self._reader = easyocr.Reader(['en'], gpu=self._gpu, verbose=False)
        return self._reader

    # ─── Public API ───

    def read_distance(self, frame) -> dict:
        """프레임에서 거리(m)를 읽는다."""
        roi = self._crop_roi(frame, self.DISTANCE_ROI)
        raw_text = self._ocr_roi(roi, allowlist='0123456789.M')
        distance_m, confidence = self._parse_distance(raw_text)

        return {
            'distance_m': distance_m,
            'raw_text': raw_text,
            'confidence': confidence,
        }

    def read_datetime(self, frame) -> dict:
        """프레임에서 촬영 일시를 읽는다."""
        roi = self._crop_roi(frame, self.DATETIME_ROI)
        raw_text = self._ocr_roi(roi, allowlist='0123456789-: ')
        dt_str, confidence = self._parse_datetime(raw_text)
        return {'datetime_str': dt_str, 'raw_text': raw_text, 'confidence': confidence}

    def read_all(self, frame) -> dict:
        """거리 + 일시 모두 읽는다."""
        dist = self.read_distance(frame)
        dt = self.read_datetime(frame)
        return {
            'distance_m': dist['distance_m'],
            'distance_raw': dist['raw_text'],
            'distance_confidence': dist['confidence'],
            'datetime_str': dt['datetime_str'],
            'datetime_raw': dt['raw_text'],
            'datetime_confidence': dt['confidence'],
        }

    def read_video_distances(self, video_path, sample_interval_frames=30,
                             progress_callback=None) -> list:
        """영상 전체에서 거리 정보를 추출한다.

        Args:
            video_path: 영상 파일 경로
            sample_interval_frames: 몇 프레임마다 OCR (기본 30 ≈ 1초)
            progress_callback: fn(frame_num, total_frames)

        Returns:
            [{'frame_number', 'distance_m', 'confidence', 'timestamp_sec'}, ...]
        """
        self._ensure_reader()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        results = []
        frame_num = 0
        last_valid = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % sample_interval_frames == 0:
                dist = self.read_distance(frame)

                if dist['distance_m'] is not None and dist['confidence'] >= 0.5:
                    d = dist['distance_m']
                    # 단조 증가 검증 + 비현실적 점프 필터 (100m 이상 점프 무시)
                    if last_valid is None or (d >= last_valid and d - last_valid < 100):
                        last_valid = d
                        results.append({
                            'frame_number': frame_num,
                            'distance_m': d,
                            'confidence': dist['confidence'],
                            'timestamp_sec': round(frame_num / fps, 2) if fps > 0 else 0,
                        })

                if progress_callback and total_frames > 0:
                    progress_callback(frame_num, total_frames)

            frame_num += 1

        cap.release()
        return results

    # ─── 내부 ───

    def _crop_roi(self, frame, roi_spec):
        h, w = frame.shape[:2]
        x1, x2 = int(w * roi_spec['x_start']), int(w * roi_spec['x_end'])
        y1, y2 = int(h * roi_spec['y_start']), int(h * roi_spec['y_end'])
        return frame[y1:y2, x1:x2].copy()

    def _ocr_roi(self, roi, allowlist=None):
        """EasyOCR로 ROI 텍스트 추출"""
        reader = self._ensure_reader()
        kwargs = {'paragraph': False}
        if allowlist:
            kwargs['allowlist'] = allowlist

        results = reader.readtext(roi, **kwargs)

        # 결과 합치기 (x좌표 순)
        results.sort(key=lambda r: r[0][0][0])
        texts = [r[1] for r in results]
        return ' '.join(texts)

    def _parse_distance(self, raw_text):
        """NNN.NNM 패턴 추출
        
        EasyOCR 출력 패턴:
          - "000.54M"       → 점 포함 인식
          - "000 54M"       → 점에서 분리
          - "000.54 M"      → M 분리
          - "000 . 54M"     → 점 별도 분리
        겵통: 앞 3자리숫자 + 뒤 2자리숫자 + M
        """
        text = raw_text.strip()
        
        # 전처리: 공백/점 정리해서 숫자만 추출
        # 모든 공백 제거 후 숫자+점+M만 남기기
        cleaned = re.sub(r'[^0-9.M]', '', text, flags=re.IGNORECASE)
        # 예: "000.54M", "00054M", "000.54M"
        
        # 패턴 1: NNN.NNM (NNN = 3자리 이상)
        m = re.search(r'(\d{3,})\.(\d{1,2})M', cleaned, re.IGNORECASE)
        if m:
            return float(f"{m.group(1)}.{m.group(2).ljust(2, '0')}"), 0.95
        
        # 패턴 2: NNNNNM (점 없이 5~6자리 + M) → 앞 3자리.2자리
        m = re.search(r'(\d{5,6})M', cleaned, re.IGNORECASE)
        if m:
            digits = m.group(1)
            # 앞에서 3자리 + 나머지
            int_part = digits[:-2]
            dec_part = digits[-2:]
            return float(f"{int_part}.{dec_part}"), 0.9

        # 패턴 3: 원문에서 공백으로 분리된 경우 "NNN NNM" → NNN.NNM
        m = re.search(r'(\d{3})\s+\.?\s*(\d{1,2})\s*M', text, re.IGNORECASE)
        if m:
            return float(f"{m.group(1)}.{m.group(2).ljust(2, '0')}"), 0.9

        # 패턴 4: 2자리 + M 만 잡힌 경우 ("54M" → 앞의 000 누락)
        # 이 경우는 부정확하므로 낮은 confidence
        m = re.search(r'(\d{1,2})\s*M', text, re.IGNORECASE)
        if m and len(m.group(1)) <= 2:
            # NNM → 0.NN으로 추정
            dec = m.group(1).ljust(2, '0')
            return float(f"0.{dec}"), 0.5

        return None, 0.0

    def _parse_datetime(self, raw_text):
        m = re.search(r'(\d{4})[.\-](\d{2})[.\-](\d{2})\s+(\d{2}):(\d{2}):(\d{2})', raw_text)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}", 0.95
        return None, 0.0


# ─── CLI ───

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python osd_ocr.py <image_or_video> [--interval N] [--no-gpu]")
        print("  image: read distance from single frame")
        print("  video: scan entire video, save .osd_distances.json")
        sys.exit(1)

    args = sys.argv[1:]
    interval = 30
    gpu = True
    files = []

    i = 0
    while i < len(args):
        if args[i] == '--interval':
            interval = int(args[i+1]); i += 2
        elif args[i] == '--no-gpu':
            gpu = False; i += 1
        else:
            files.append(args[i]); i += 1

    reader = OSDDistanceReader(gpu=gpu, debug=True)
    video_exts = {'.avi', '.mp4', '.mkv', '.mov'}

    for path in files:
        ext = os.path.splitext(path)[1].lower()

        if ext in video_exts:
            print(f"\nScanning: {os.path.basename(path)} (interval={interval})")
            t0 = time.time()

            def progress(fn, total):
                if fn % (interval * 10) == 0:
                    elapsed = time.time() - t0
                    pct = fn / total * 100
                    print(f"  {fn}/{total} ({pct:.0f}%) {elapsed:.1f}s")

            results = reader.read_video_distances(path, interval, progress)
            elapsed = time.time() - t0

            print(f"\n=== {os.path.basename(path)} ===")
            print(f"  Readings: {len(results)}")
            print(f"  Time: {elapsed:.1f}s")
            if results:
                print(f"  First: {results[0]['distance_m']:.2f}m @ frame {results[0]['frame_number']}")
                print(f"  Last:  {results[-1]['distance_m']:.2f}m @ frame {results[-1]['frame_number']}")
                total_dist = results[-1]['distance_m'] - results[0]['distance_m']
                print(f"  Span:  {total_dist:.2f}m")

                out = path + '.osd_distances.json'
                with open(out, 'w') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"  Saved: {out}")
        else:
            frame = cv2.imread(path)
            if frame is None:
                print(f"Cannot read: {path}")
                continue
            result = reader.read_all(frame)
            name = os.path.basename(path)
            print(f"\n{name}:")
            print(f"  Distance: {result['distance_m']}m  (conf={result['distance_confidence']:.2f})")
            print(f"  Raw: '{result['distance_raw']}'")
            if result['datetime_str']:
                print(f"  DateTime: {result['datetime_str']}")


if __name__ == '__main__':
    main()
