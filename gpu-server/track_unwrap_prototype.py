#!/usr/bin/env python3
"""
track_unwrap_prototype.py — 결함 추적 기반 전개도/구간 통합 면적비 프로토타입

목적(실험): OSD 거리에 의존하지 않고, PPNet 전개(θ×z) 공간에서
  1) 프레임 간 z 변위를 위상상관(phase correlation)으로 추정 → 모션 정합
  2) 결함을 전개공간 IoU로 추적 → 물리적 결함 단위 식별(중복 제거)
  3) 모션 정합 모자이크에 가시/결함 누적 → 통합 면적비 산출
하여 기존 stop+OSD 방식의 한계(OSD 없으면 dedup 불가)를 보완할 수 있는지 검증한다.

기존 survey 경로(pipe_survey.py / api.py)는 건드리지 않는 독립 실험 스크립트.

사용:
  ../.venv/bin/python track_unwrap_prototype.py <video.mp4> \
      --pipe-mm 300 --stride 5 --max-frames 800 --ppm 1.0 \
      --conf 0.25 --iou 0.7 --imgsz 640

출력: runs_track_proto/<videoname>/  (mosaic.jpg, tracks.jpg, summary.json)
"""
import os
import sys
import json
import argparse
import numpy as np
import cv2


# ──────────────────────────────────────────────────────────────────────
def build_engine_and_mapper(ppnet_path, frame_shape, pipe_mm, ppm, max_depth_mm):
    """GNUMappingEngine(PPNet 싱글톤) + 해상도별 PipeMapper 생성."""
    from gnu_mapping import GNUMappingEngine, PipeMapper, CAMERA_PARAMS, detect_resolution
    eng = GNUMappingEngine(ppnet_path, pipe_diameter_mm=int(pipe_mm),
                           pixel_per_mm=ppm, max_depth_mm=max_depth_mm)
    h, w = frame_shape[:2]
    cam = dict(CAMERA_PARAMS[detect_resolution(w, h)])
    mapper = PipeMapper(f=cam['f'], cx=cam['cx'], cy=cam['cy'],
                        pipe_diameter_mm=int(pipe_mm), water=False,
                        pixel_per_mm=ppm, max_depth_mm=max_depth_mm)
    return eng, mapper


def yolo_instance_masks(result, w, h):
    """ultralytics 결과 → [{label, conf, mask(H,W uint8)}] (인스턴스 마스크)."""
    out = []
    if result.masks is None:
        return out
    names = result.names
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    polys = result.masks.xy  # 원본 좌표 폴리곤 리스트
    for i, poly in enumerate(polys):
        if poly is None or len(poly) < 3:
            continue
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [poly.astype(np.int32)], 1)
        out.append({
            'label': names[int(cls[i])] if int(cls[i]) in names else str(cls[i]),
            'conf': float(conf[i]),
            'mask': m,
        })
    return out


def bbox_of(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]  # x0,y0,x1,y1


def iou_box(a, b):
    """글로벌 좌표 bbox IoU (a,b = [x0,y0,x1,y1])."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def phase_shift_z(prev_gray, cur_gray, hann):
    """위상상관으로 cur→prev 정렬 시 (dx, dy) 추정. dy = z(축) 변위 px, resp=신뢰도."""
    try:
        (dx, dy), resp = cv2.phaseCorrelate(prev_gray, cur_gray, hann)
        return float(dx), float(dy), float(resp)
    except Exception:
        return 0.0, 0.0, 0.0


# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--pipe-mm', type=float, default=300)
    ap.add_argument('--stride', type=int, default=5, help='N프레임마다 샘플')
    ap.add_argument('--max-frames', type=int, default=800, help='최대 샘플 프레임 수')
    ap.add_argument('--ppm', type=float, default=1.0, help='전개 pixel_per_mm (면적비용 저해상)')
    ap.add_argument('--max-depth-mm', type=float, default=300)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.7)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--track-iou', type=float, default=0.2, help='트랙 매칭 IoU 임계')
    ap.add_argument('--yolo', default=None, help='YOLO 가중치 (기본: ../yolo_best.pt)')
    ap.add_argument('--ppnet', default=None, help='PPNet 가중치 (기본: weights/ppnet.pt)')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ppnet_path = args.ppnet or os.path.join(here, 'weights', 'ppnet.pt')
    yolo_path = args.yolo or os.path.join(os.path.dirname(here), 'yolo_best.pt')
    vname = os.path.splitext(os.path.basename(args.video))[0]
    outdir = args.outdir or os.path.join(here, 'runs_track_proto', vname)
    os.makedirs(outdir, exist_ok=True)

    for p in (args.video, ppnet_path, yolo_path):
        if not os.path.exists(p):
            print(f'[ERR] not found: {p}'); sys.exit(1)

    print(f'[proto] video={args.video}')
    print(f'[proto] stride={args.stride} max_frames={args.max_frames} ppm={args.ppm} '
          f'conf={args.conf} iou={args.iou} imgsz={args.imgsz}')

    from ultralytics import YOLO
    yolo = YOLO(yolo_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print('[ERR] cannot open video'); sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ret, frame0 = cap.read()
    if not ret:
        print('[ERR] cannot read first frame'); sys.exit(1)

    eng, mapper = build_engine_and_mapper(ppnet_path, frame0.shape, args.pipe_mm,
                                          args.ppm, args.max_depth_mm)
    out_h, out_w = mapper.out_h, mapper.out_w
    hann = cv2.createHanningWindow((out_w, out_h), cv2.CV_32F)
    print(f'[proto] unwrap canvas per-frame: {out_h}x{out_w} (max_depth={args.max_depth_mm}mm @ {args.ppm}px/mm)')

    # ── Phase A: 프레임별 전개 + 검출 + 위상상관 z변위 ──
    frames = []          # [{idx, vis(bool out_h×out_w), dets:[{label,conf,uw_mask,lbbox}], dz}]
    prev_gray = None
    cum_offsets = []     # 프레임별 누적 z offset(px)
    cum = 0.0
    sampled = 0
    fidx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while sampled < args.max_frames:
        ok = cap.grab()
        if not ok:
            break
        if fidx % args.stride != 0:
            fidx += 1
            continue
        ret, frame = cap.retrieve()
        fidx += 1
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            pose = eng.ppnet.run(rgb)['pose']
        except Exception as e:
            print(f'  [skip {fidx}] ppnet: {e}'); continue
        unwrapped = mapper.unwrap(rgb, pose)              # (out_h,out_w,3)
        vis = np.any(unwrapped > 0, axis=-1)
        gray = cv2.cvtColor(unwrapped, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # 위상상관 z변위
        dz = 0.0
        if prev_gray is not None:
            _, dy, resp = phase_shift_z(prev_gray, gray, hann)
            # 비정상 변위(전개 높이의 절반 초과)·저신뢰는 0 처리
            if resp < 0.05 or abs(dy) > out_h * 0.5:
                dy = 0.0
            dz = dy
        prev_gray = gray
        cum += dz
        cum_offsets.append(cum)

        # YOLO 인스턴스 검출 → 전개공간 마스크
        res = yolo(rgb, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                   verbose=False)[0]
        inst = yolo_instance_masks(res, frame.shape[1], frame.shape[0])
        dets = []
        for d in inst:
            uw = mapper.unwrap(d['mask'], pose)
            uwb = (np.any(uw > 0, axis=-1) if uw.ndim == 3 else (uw > 0)) & vis
            if int(uwb.sum()) == 0:
                continue
            lb = bbox_of(uwb)
            dets.append({'label': d['label'], 'conf': d['conf'],
                         'uw_mask': uwb, 'lbbox': lb})

        frames.append({'idx': fidx, 'vis': vis, 'dets': dets, 'dz': dz})
        sampled += 1
        if sampled % 20 == 0:
            print(f'  sampled {sampled} (frame {fidx}/{total}) cum_z={cum:.0f}px dets={len(dets)}')
    cap.release()

    if not frames:
        print('[ERR] no frames processed'); sys.exit(1)
    print(f'[proto] sampled {len(frames)} frames')

    # 누적 offset 정규화(최소=0) → 글로벌 z px
    base = min(cum_offsets)
    g_off = [int(round(c - base)) for c in cum_offsets]
    canvas_rows = max(g_off) + out_h
    z_span_mm = (max(g_off)) / args.ppm
    print(f'[proto] estimated z-span (phase-corr) = {z_span_mm:.0f}mm ({z_span_mm/1000:.1f}m), '
          f'canvas {canvas_rows}x{out_w}')

    # ── Phase B: 모자이크 누적(모션 정합) + naive 합산 ──
    mosaic_vis = np.zeros((canvas_rows, out_w), dtype=np.uint8)
    mosaic_def = np.zeros((canvas_rows, out_w), dtype=np.uint8)
    raw_vis = raw_def = 0
    for fr, off in zip(frames, g_off):
        sl = slice(off, off + out_h)
        v = fr['vis'].astype(np.uint8)
        mosaic_vis[sl] |= v
        raw_vis += int(v.sum())
        for d in fr['dets']:
            dm = d['uw_mask'].astype(np.uint8)
            mosaic_def[sl] |= dm
            raw_def += int(dm.sum())
    mosaic_def &= mosaic_vis
    vis_cells = int(mosaic_vis.sum())
    def_cells = int(mosaic_def.sum())
    ratio_mosaic = round(def_cells / vis_cells * 100, 3) if vis_cells else 0.0
    ratio_naive = round(raw_def / raw_vis * 100, 3) if raw_vis else 0.0
    overlap = round(raw_vis / vis_cells, 2) if vis_cells else None

    # ── Phase C: 전개공간 IoU 추적 (글로벌 좌표) ──
    tracks = []   # {id,label,gbox,best_area,best_frame,nframes,last_seen}
    next_id = 0
    for fi, (fr, off) in enumerate(zip(frames, g_off)):
        cur = []
        for d in fr['dets']:
            lb = d['lbbox']
            gb = [lb[0], lb[1] + off, lb[2], lb[3] + off]   # 글로벌 bbox
            area = int(d['uw_mask'].sum())
            cur.append({'label': d['label'], 'gbox': gb, 'area': area})
        # 활성 트랙(최근 관측)과 IoU 매칭
        active = [t for t in tracks if fi - t['last_seen'] <= 3]
        used = set()
        for c in cur:
            best, bi = 0.0, -1
            for ti, t in enumerate(active):
                if t['label'] != c['label'] or ti in used:
                    continue
                j = iou_box(c['gbox'], t['gbox'])
                if j > best:
                    best, bi = j, ti
            if best >= args.track_iou and bi >= 0:
                t = active[bi]; used.add(bi)
                t['last_seen'] = fi; t['nframes'] += 1
                # 글로벌 bbox 갱신(이동평균) + 최대 면적 뷰 기록
                t['gbox'] = [int((t['gbox'][k] + c['gbox'][k]) / 2) for k in range(4)]
                if c['area'] > t['best_area']:
                    t['best_area'] = c['area']; t['best_frame'] = fr['idx']
            else:
                tracks.append({'id': next_id, 'label': c['label'], 'gbox': c['gbox'],
                               'best_area': c['area'], 'best_frame': fr['idx'],
                               'nframes': 1, 'last_seen': fi})
                next_id += 1

    # 단발성(1프레임) 트랙은 잡음일 가능성 → 별도 집계
    solid = [t for t in tracks if t['nframes'] >= 2]
    by_label = {}
    for t in tracks:
        by_label.setdefault(t['label'], {'tracks': 0, 'solid': 0, 'best_area_mm2_sum': 0.0})
        by_label[t['label']]['tracks'] += 1
        if t['nframes'] >= 2:
            by_label[t['label']]['solid'] += 1
        by_label[t['label']]['best_area_mm2_sum'] += round(t['best_area'] / (args.ppm ** 2), 1)

    # 트랙 기반 면적합(각 결함 1회, 최대뷰) / 모자이크 가시 → 대안 면적비
    track_def_area = sum(t['best_area'] for t in solid)
    ratio_track = round(track_def_area / vis_cells * 100, 3) if vis_cells else 0.0

    # ── 출력 ──
    viz = np.zeros((canvas_rows, out_w, 3), dtype=np.uint8)
    viz[mosaic_vis > 0] = (70, 70, 70)
    viz[mosaic_def > 0] = (40, 40, 220)
    cv2.imwrite(os.path.join(outdir, 'mosaic.jpg'), viz, [cv2.IMWRITE_JPEG_QUALITY, 88])

    # 트랙 박스 오버레이
    tviz = viz.copy()
    for t in solid:
        x0, y0, x1, y1 = t['gbox']
        col = (0, 140, 255) if 'nodule' in t['label'] else (255, 130, 80)
        cv2.rectangle(tviz, (x0, y0), (x1, y1), col, 1)
        cv2.putText(tviz, f"#{t['id']}", (x0, max(10, y0 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(outdir, 'tracks.jpg'), tviz, [cv2.IMWRITE_JPEG_QUALITY, 88])

    summary = {
        'video': args.video, 'fps': fps, 'total_frames': total,
        'sampled_frames': len(frames), 'stride': args.stride, 'ppm': args.ppm,
        'max_depth_mm': args.max_depth_mm,
        'infer': {'conf': args.conf, 'iou': args.iou, 'imgsz': args.imgsz},
        'z_registration': 'phase_correlation(unwrap-space)',
        'z_span_mm_estimated': round(z_span_mm, 1),
        'canvas': [canvas_rows, out_w],
        'area_ratio_pct': {
            'mosaic_dedup': ratio_mosaic,     # 모션 정합 모자이크(중복 제거)
            'naive_sum': ratio_naive,         # 프레임 단순 합(중복 포함)
            'track_best_view': ratio_track,   # 트랙별 최대뷰 합 / 모자이크 가시
        },
        'overlap_factor': overlap,
        'visible_cells': vis_cells, 'defect_cells': def_cells,
        'tracks_total': len(tracks),
        'tracks_solid(>=2frames)': len(solid),
        'by_label': by_label,
        'tracks': [{'id': t['id'], 'label': t['label'], 'nframes': t['nframes'],
                    'best_area_mm2': round(t['best_area'] / (args.ppm ** 2), 1),
                    'best_frame': t['best_frame']} for t in sorted(solid, key=lambda x: -x['best_area'])[:30]],
    }
    with open(os.path.join(outdir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('\n===== RESULT =====')
    print(f'sampled frames     : {len(frames)}')
    print(f'z-span (phase-corr): {z_span_mm:.0f}mm  (overlap factor {overlap})')
    print(f'distinct defects   : total={len(tracks)}  solid(>=2f)={len(solid)}  {dict((k,v["solid"]) for k,v in by_label.items())}')
    print(f'area ratio  mosaic-dedup = {ratio_mosaic}%   naive-sum = {ratio_naive}%   track-bestview = {ratio_track}%')
    print(f'outputs -> {outdir}/  (mosaic.jpg, tracks.jpg, summary.json)')


if __name__ == '__main__':
    main()
