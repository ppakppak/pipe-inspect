#!/usr/bin/env python3
"""
OSD 샘플 영상 특성 분석 스크립트

목적: Global Area Ratio 설계 검증을 위한 선행 분석.
- OSD 거리값 시계열
- OCR 신뢰도
- 정지/이동 상태 (SSIM / Optical Flow 기반 교차검증)
- 정지 구간 길이 분포
- 정지 구간 사이 Δz 분포
- 정지 구간 내 선명도 / 상호 일관성

출력: 지정 디렉토리에 PNG 플롯 + JSON 요약.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from osd_ocr import OSDDistanceReader


def laplacian_sharpness(frame_gray):
    return float(cv2.Laplacian(frame_gray, cv2.CV_64F).var())


def compute_flow_magnitude(prev_gray, cur_gray, exclude_center_ratio=0.25):
    """Farneback dense flow. VP 주변 중심 영역 제외한 median magnitude."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=3, winsize=21, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    h, w = cur_gray.shape
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    # 중심 VP 영역(파이프 수렴점) 제외
    ch, cw = int(h * exclude_center_ratio), int(w * exclude_center_ratio)
    y0, y1 = (h - ch) // 2, (h + ch) // 2
    x0, x1 = (w - cw) // 2, (w + cw) // 2
    mask = np.ones_like(mag, dtype=bool)
    mask[y0:y1, x0:x1] = False
    return float(np.median(mag[mask]))


def compute_ssim_fast(prev_gray, cur_gray, size=256):
    """빠른 SSIM 근사 (다운샘플 후 상관계수)."""
    a = cv2.resize(prev_gray, (size, size))
    b = cv2.resize(cur_gray, (size, size))
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    denom = (np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-8
    return float((a * b).sum() / denom)


def clean_ocr_sequence(distances, context_tolerance_m=2.0):
    """OCR 오류 후처리.

    규칙:
    1. z > 100 이고 z/10 이 인접 유효값과 ±context_tolerance_m 이내면 z/10 로 교체
    2. 나머지 이상치: |z - 이웃 중앙값| > context_tolerance_m*5 이면 None 으로
    """
    n = len(distances)
    # 1. 10배 오인식 보정
    corrected = [dict(d) for d in distances]
    # 유효값 인덱스만 모아서 빠른 이웃 찾기
    for i, d in enumerate(corrected):
        z = d['distance_m']
        if z is None or z <= 100:
            continue
        # 앞뒤 유효값 탐색 (10배 오인식 제외)
        neighbors = []
        for j in range(max(0, i - 20), min(n, i + 20)):
            if j == i:
                continue
            zj = corrected[j]['distance_m']
            if zj is not None and zj <= 100:
                neighbors.append(zj)
        if not neighbors:
            continue
        median = float(np.median(neighbors))
        if abs(z / 10 - median) <= context_tolerance_m:
            corrected[i]['distance_m'] = z / 10
            corrected[i]['corrected'] = '10x_divided'
        elif z > 100:
            # 보정해도 안 맞으면 제거
            corrected[i]['distance_m'] = None
            corrected[i]['corrected'] = 'dropped_high'

    # 2. 잔여 outlier 제거 (이웃 중앙값 대비 과도)
    for i, d in enumerate(corrected):
        z = d['distance_m']
        if z is None:
            continue
        neighbors = []
        for j in range(max(0, i - 10), min(n, i + 10)):
            if j == i:
                continue
            zj = corrected[j]['distance_m']
            if zj is not None:
                neighbors.append(zj)
        if len(neighbors) < 3:
            continue
        median = float(np.median(neighbors))
        if abs(z - median) > context_tolerance_m * 5:
            corrected[i]['distance_m'] = None
            corrected[i]['corrected'] = 'dropped_outlier'

    return corrected


def interpolate_gaps(distances):
    """OCR 실패 프레임을 전후 유효값 시간가중 선형보간."""
    n = len(distances)
    out = [dict(d) for d in distances]
    # 각 프레임의 가장 가까운 이전/이후 유효값 인덱스
    prev_valid = [-1] * n
    cur = -1
    for i in range(n):
        if out[i]['distance_m'] is not None:
            cur = i
        prev_valid[i] = cur
    next_valid = [-1] * n
    cur = -1
    for i in range(n - 1, -1, -1):
        if out[i]['distance_m'] is not None:
            cur = i
        next_valid[i] = cur

    for i in range(n):
        if out[i]['distance_m'] is not None:
            continue
        p = prev_valid[i]
        q = next_valid[i]
        if p >= 0 and q >= 0 and p != q:
            zp = out[p]['distance_m']
            zq = out[q]['distance_m']
            w = (i - p) / (q - p)
            out[i]['distance_m'] = zp + (zq - zp) * w
            out[i]['interpolated'] = True
        elif p >= 0:
            out[i]['distance_m'] = out[p]['distance_m']
            out[i]['interpolated'] = 'from_prev'
        elif q >= 0:
            out[i]['distance_m'] = out[q]['distance_m']
            out[i]['interpolated'] = 'from_next'
    return out


def segment_osd_constant_runs(distances):
    """OSD 값이 일정한 구간을 [(start_idx, end_idx, z), ...] 로 분할."""
    runs = []
    if not distances:
        return runs
    cur_z = distances[0]['distance_m']
    start = 0
    for i in range(1, len(distances)):
        z = distances[i]['distance_m']
        if z is None or cur_z is None:
            if z != cur_z:
                runs.append((start, i - 1, cur_z))
                cur_z = z
                start = i
            continue
        if abs(z - cur_z) > 1e-4:
            runs.append((start, i - 1, cur_z))
            cur_z = z
            start = i
    runs.append((start, len(distances) - 1, cur_z))
    return runs


def analyze(video_path, out_dir, ocr_stride=5, motion_stride=2, max_frames=None, gpu=True,
            abs_flow_static_threshold=0.5, direction='auto'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = min(total, max_frames) if max_frames else total

    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] {w}x{h} @ {fps:.2f}fps, {total} frames (analyzing {n_frames})")

    reader = OSDDistanceReader(gpu=gpu, debug=False)

    # Pass 1: OSD OCR + per-frame motion (순차 처리, prev_gray 유지)
    distances = []       # 모든 프레임 (OCR은 stride마다, 나머지는 None)
    sharpness_log = []
    flow_log = []
    ssim_log = []

    prev_gray = None
    t0 = time.time()
    frame_idx = 0

    while frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = laplacian_sharpness(gray)
        sharpness_log.append(sharp)

        if prev_gray is not None and frame_idx % motion_stride == 0:
            flow_med = compute_flow_magnitude(prev_gray, gray)
            ssim_val = compute_ssim_fast(prev_gray, gray)
        else:
            flow_med = None
            ssim_val = None
        flow_log.append(flow_med)
        ssim_log.append(ssim_val)

        if frame_idx % ocr_stride == 0:
            res = reader.read_distance(frame)
            distances.append({
                'frame': frame_idx,
                'distance_m': res['distance_m'],
                'raw_text': res['raw_text'],
                'confidence': res['confidence'],
            })
        else:
            distances.append({
                'frame': frame_idx,
                'distance_m': None,
                'raw_text': None,
                'confidence': None,
            })

        prev_gray = gray
        frame_idx += 1

        if frame_idx % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / frame_idx * (n_frames - frame_idx)
            print(f"  [{frame_idx}/{n_frames}] elapsed={elapsed:.1f}s eta={eta:.1f}s")

    cap.release()
    print(f"[INFO] Pass 1 done in {time.time() - t0:.1f}s")

    # OCR 원본 집계
    raw_ocr = [d for d in distances if d['distance_m'] is not None]
    ocr_total = sum(1 for d in distances if d['frame'] % ocr_stride == 0)
    ocr_raw_success = len(raw_ocr)

    # OCR 후처리: 10배 오인식 보정 + outlier 제거
    cleaned = clean_ocr_sequence(distances)
    # 실패 프레임 보간
    filled_tmp = interpolate_gaps(cleaned)

    ocr_success = sum(1 for d in cleaned if d['distance_m'] is not None)
    corrected_10x = sum(1 for d in cleaned if d.get('corrected') == '10x_divided')
    dropped = sum(1 for d in cleaned if d.get('corrected') in ('dropped_high', 'dropped_outlier'))

    # filled dict 재구성 (이후 코드 호환 형식)
    filled = []
    for d in filled_tmp:
        filled.append({
            'frame': d['frame'],
            'distance_m': d['distance_m'],
            'confidence': d['confidence'] or 0.0,
            'is_raw': cleaned[d['frame'] if d['frame'] < len(cleaned) else 0]['distance_m'] is not None
                      if d.get('interpolated') is None else False,
        })

    # 정지 구간 분할
    runs = segment_osd_constant_runs(filled)
    # 이동 구간(= 한 run에서 다음 run으로 넘어가는 경계): 이 설계에서는
    # 같은 z가 유지되는 run 자체를 "정지 후보"로 봄.

    # Δz 분포 (연속 run 사이)
    dz_values = []
    for i in range(1, len(runs)):
        z_prev = runs[i - 1][2]
        z_cur = runs[i][2]
        if z_prev is not None and z_cur is not None:
            dz_values.append(z_cur - z_prev)

    # run 길이 분포
    run_lengths = [(e - s + 1) for s, e, _ in runs]

    # 정지/이동 판정 (flow 기준): 낮은 flow = 정지
    flow_arr = np.array([f if f is not None else np.nan for f in flow_log], dtype=np.float64)
    ssim_arr = np.array([s if s is not None else np.nan for s in ssim_log], dtype=np.float64)
    flow_threshold = float(abs_flow_static_threshold)
    is_static = (flow_arr < flow_threshold)  # NaN은 False

    # 정지 구간 내 선명도 통계
    run_stats = []
    for s, e, z in runs:
        sharps = sharpness_log[s:e + 1]
        flows_in = [f for f in flow_log[s:e + 1] if f is not None]
        run_stats.append({
            'start': s,
            'end': e,
            'length': e - s + 1,
            'z_m': z,
            'sharp_mean': float(np.mean(sharps)) if sharps else None,
            'sharp_max': float(np.max(sharps)) if sharps else None,
            'flow_median': float(np.median(flows_in)) if flows_in else None,
        })

    # ===== 시각화 =====
    # 1. OSD z 시계열 (원본 OCR + forward-fill 비교)
    fig, ax = plt.subplots(figsize=(14, 4))
    frames_axis = np.arange(len(filled))
    filled_z = np.array([d['distance_m'] if d['distance_m'] is not None else np.nan for d in filled])
    raw_mask = np.array([d['is_raw'] for d in filled])
    ax.plot(frames_axis, filled_z, '-', color='steelblue', linewidth=0.8, label='forward-fill')
    ax.scatter(frames_axis[raw_mask], filled_z[raw_mask], s=8, color='orange', label='OCR raw')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Distance (m)')
    ax.set_title('OSD Distance Timeline')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / '01_osd_timeline.png', dpi=110)
    plt.close()

    # 2. OCR 신뢰도
    fig, ax = plt.subplots(figsize=(14, 3))
    confs = [d['confidence'] if d['confidence'] is not None else np.nan for d in distances]
    ax.plot(confs, '-', linewidth=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('OCR confidence')
    ax.set_title(f'OCR Confidence  (success: {ocr_success}/{ocr_total} = {100*ocr_success/max(1,ocr_total):.1f}%)')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / '02_ocr_confidence.png', dpi=110)
    plt.close()

    # 3. 정지/이동 오버레이
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(frames_axis, filled_z, '-', color='steelblue', linewidth=0.8)
    # OSD run 경계 표시
    for s, e, z in runs:
        ax1.axvspan(s, e, alpha=0.08, color='green')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('OSD z + Run Segments (green bands)')
    ax1.grid(alpha=0.3)

    ax2.plot(flow_arr, '-', color='darkred', linewidth=0.7, label='flow median (non-center)')
    ax2.axhline(flow_threshold, color='black', linestyle='--', linewidth=0.8, label=f'static threshold={flow_threshold:.2f}')
    ax2.fill_between(frames_axis, 0, flow_arr, where=is_static, alpha=0.15, color='green', label='static (flow low)')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Flow magnitude')
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / '03_static_vs_moving.png', dpi=110)
    plt.close()

    # 4. run 길이 분포
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(run_lengths, bins=40, color='steelblue', edgecolor='black')
    ax.set_xlabel('Run length (frames, same OSD z)')
    ax.set_ylabel('Count')
    ax.set_title(f'OSD Constant Run Lengths  (n_runs={len(runs)})')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / '04_run_length_hist.png', dpi=110)
    plt.close()

    # 5. Δz 히스토그램
    fig, ax = plt.subplots(figsize=(9, 4))
    if dz_values:
        ax.hist([v * 1000 for v in dz_values], bins=50, color='coral', edgecolor='black')
        ax.set_xlabel('Δz between consecutive runs (mm)')
        ax.set_ylabel('Count')
        ax.set_title(f'Δz Distribution  (runs transitions={len(dz_values)}, max_depth_mm=300 ref)')
        ax.axvline(300, color='red', linestyle='--', linewidth=1.0, label='current unwrap max_depth=300mm')
        ax.axvline(-300, color='red', linestyle='--', linewidth=1.0)
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / '05_dz_hist.png', dpi=110)
    plt.close()

    # 6. 정지 구간별 선명도 분포
    fig, ax = plt.subplots(figsize=(14, 4))
    sharp_arr = np.array(sharpness_log)
    ax.plot(sharp_arr, '-', linewidth=0.6, color='gray', label='per-frame Laplacian var')
    # run 별 평균 점
    for rs in run_stats:
        if rs['sharp_mean'] is not None:
            ax.scatter((rs['start'] + rs['end']) / 2, rs['sharp_mean'],
                       s=20, color='red', alpha=0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Laplacian variance')
    ax.set_title('Sharpness (per frame, red dots = run means)')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / '06_sharpness.png', dpi=110)
    plt.close()

    # 7. 샘플 run 내 SSIM matrix (가장 긴 run)
    if runs:
        longest = max(runs, key=lambda r: r[1] - r[0])
        s, e, z = longest
        # 샘플 최대 20프레임
        indices = np.linspace(s, e, min(20, e - s + 1), dtype=int)
        cap2 = cv2.VideoCapture(video_path)
        grays = []
        for idx in indices:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
            r2, f2 = cap2.read()
            if r2:
                grays.append(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
        cap2.release()
        n = len(grays)
        M = np.ones((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                v = compute_ssim_fast(grays[i], grays[j])
                M[i, j] = M[j, i] = v
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, cmap='viridis', vmin=0, vmax=1)
        ax.set_title(f'SSIM matrix inside longest run\nframes {s}~{e}, z={z}m (n={n} samples)')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(out_dir / '07_ssim_longest_run.png', dpi=110)
        plt.close()

    # ===== 요약 JSON =====
    summary = {
        'video_path': str(video_path),
        'video_info': {'width': w, 'height': h, 'fps': fps, 'total_frames': total, 'analyzed_frames': n_frames},
        'ocr_stride': ocr_stride,
        'motion_stride': motion_stride,
        'ocr': {
            'attempted': ocr_total,
            'raw_success': ocr_raw_success,
            'raw_success_rate': ocr_raw_success / max(1, ocr_total),
            'after_cleanup_success': ocr_success,
            'corrected_10x': corrected_10x,
            'dropped': dropped,
        },
        'direction': direction,
        'osd_runs': {
            'count': len(runs),
            'length_stats': {
                'mean': float(np.mean(run_lengths)),
                'median': float(np.median(run_lengths)),
                'max': int(np.max(run_lengths)),
                'min': int(np.min(run_lengths)),
            },
        },
        'dz_mm': {
            'count': len(dz_values),
            'mean_mm': float(np.mean(dz_values) * 1000) if dz_values else None,
            'median_mm': float(np.median(dz_values) * 1000) if dz_values else None,
            'max_mm': float(np.max(np.abs(dz_values)) * 1000) if dz_values else None,
            'negative_count': sum(1 for v in dz_values if v < 0),
        },
        'flow_static_threshold': flow_threshold,
        'static_ratio_by_flow': float(np.sum(is_static) / max(1, len(is_static))),
        'z_range': {
            'first_valid': raw_ocr[0]['distance_m'] if raw_ocr else None,
            'last_valid': raw_ocr[-1]['distance_m'] if raw_ocr else None,
            'cleaned_first': next((d['distance_m'] for d in cleaned if d['distance_m'] is not None), None),
            'cleaned_last': next((d['distance_m'] for d in reversed(cleaned) if d['distance_m'] is not None), None),
        },
    }

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / 'runs.json').write_text(json.dumps(run_stats, indent=2, ensure_ascii=False))

    print('\n==== SUMMARY ====')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\n[OK] Outputs in {out_dir}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ocr-stride', type=int, default=5, help='OCR every N frames (default 5)')
    ap.add_argument('--motion-stride', type=int, default=2)
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--flow-static-threshold', type=float, default=0.5)
    ap.add_argument('--direction', default='auto', choices=['auto', 'forward', 'reverse'])
    args = ap.parse_args()

    analyze(
        video_path=args.video,
        out_dir=args.out,
        ocr_stride=args.ocr_stride,
        motion_stride=args.motion_stride,
        max_frames=args.max_frames,
        gpu=not args.cpu,
        abs_flow_static_threshold=args.flow_static_threshold,
        direction=args.direction,
    )
