#!/usr/bin/env python3
"""OCR raw 출력 패턴 샘플링 — Δz 이상치의 원인 파악용."""

import argparse
import os
import sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osd_ocr import OSDDistanceReader


def probe(video_path, n_samples=120, stride=5, gpu=True):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reader = OSDDistanceReader(gpu=gpu, debug=False)

    print(f"[INFO] total frames: {total}")
    print(f"[INFO] sampling stride={stride}, max_samples={n_samples}\n")
    print(f"{'frame':>6}  {'raw_text':<16}  {'parsed_m':>10}  {'conf':>5}")
    print('-' * 50)

    idx = 0
    shown = 0
    while shown < n_samples:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            res = reader.read_distance(frame)
            raw = (res['raw_text'] or '').replace('\n', ' ')[:16]
            d = res['distance_m']
            c = res['confidence'] or 0
            d_str = f"{d:.3f}" if d is not None else "None"
            print(f"{idx:>6}  {raw:<16}  {d_str:>10}  {c:>5.2f}")
            shown += 1
        idx += 1

    cap.release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--n', type=int, default=120)
    ap.add_argument('--stride', type=int, default=5)
    args = ap.parse_args()
    probe(args.video, n_samples=args.n, stride=args.stride)
