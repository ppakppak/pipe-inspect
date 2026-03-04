#!/usr/bin/env python3
"""Nightly frame cache warmer for pipe-inspector-staging.

- Scans all project annotations
- Collects annotated frame numbers
- Extracts only missing frames into frame_cache
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import cv2


def to_web_video_path(src_path: str, videos_web_dir: Path) -> Path:
    p = Path(src_path)
    s = str(p)

    if 'SAHARA' in s:
        parts = list(p.parts)
        i = parts.index('SAHARA')
        rel = Path(*parts[i + 1 :])
        return (videos_web_dir / 'SAHARA' / rel).with_suffix('.mp4')

    if '관내시경영상' in s:
        parts = list(p.parts)
        i = parts.index('관내시경영상')
        rel = Path(*parts[i + 1 :])
        return (videos_web_dir / '관내시경영상' / rel).with_suffix('.mp4')

    return Path(s.replace('.avi', '.mp4').replace('.AVI', '.mp4'))


def cache_path(cache_root: Path, video_path: str, frame_num: int) -> Path:
    key = hashlib.sha1(video_path.encode('utf-8')).hexdigest()
    return cache_root / key[:2] / key / f"{int(frame_num):06d}.jpg"


def collect_tasks(base_projects_dir: Path, videos_web_dir: Path) -> dict[str, set[int]]:
    tasks: dict[str, set[int]] = defaultdict(set)

    user_dirs = [d for d in base_projects_dir.iterdir() if d.is_dir()]
    for user_dir in sorted(user_dirs):
        for project_dir in sorted([d for d in user_dir.iterdir() if d.is_dir()]):
            project_json = project_dir / 'project.json'
            annotations_dir = project_dir / 'annotations'
            if not project_json.exists() or not annotations_dir.exists():
                continue

            try:
                pj = json.loads(project_json.read_text())
            except Exception:
                continue

            video_map = {}
            for v in pj.get('videos', []):
                vid = v.get('video_id')
                vpath = v.get('video_path')
                if vid and vpath:
                    video_map[vid] = str(to_web_video_path(vpath, videos_web_dir))

            for video_anno_dir in sorted([d for d in annotations_dir.iterdir() if d.is_dir()]):
                video_id = video_anno_dir.name
                vpath = video_map.get(video_id)
                if not vpath or not Path(vpath).exists():
                    continue

                for jf in video_anno_dir.glob('*.json'):
                    name = jf.name
                    if 'backup' in name or 'before_fix' in name or 'discussions' in name:
                        continue
                    try:
                        data = json.loads(jf.read_text())
                    except Exception:
                        continue

                    annos = data.get('annotations', {})
                    if not isinstance(annos, dict):
                        continue

                    for fnum, items in annos.items():
                        if not isinstance(items, list) or not items:
                            continue
                        has_polygon = False
                        for it in items:
                            poly = it.get('polygon') if isinstance(it, dict) else None
                            if isinstance(poly, list) and len(poly) >= 3:
                                has_polygon = True
                                break
                        if not has_polygon:
                            continue
                        try:
                            tasks[vpath].add(int(fnum))
                        except Exception:
                            continue

    return tasks


def warm_one_video(video_path: str, frames: set[int], cache_root: Path) -> tuple[int, int, int]:
    frames_sorted = sorted(frames)
    if not frames_sorted:
        return 0, 0, 0

    missing = [f for f in frames_sorted if not cache_path(cache_root, video_path, f).exists()]
    if not missing:
        return len(frames_sorted), 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return len(frames_sorted), 0, len(missing)

    created = 0
    failed = 0
    try:
        for fnum in missing:
            cp = cache_path(cache_root, video_path, fnum)
            cp.parent.mkdir(parents=True, exist_ok=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            if not ret or frame is None:
                failed += 1
                continue

            ok = cv2.imwrite(str(cp), frame)
            if ok:
                created += 1
            else:
                failed += 1
    finally:
        cap.release()

    return len(frames_sorted), created, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-projects-dir', default='/home/intu/Nas2/k_water/pipe_inspector_data')
    ap.add_argument('--videos-web-dir', default='/home/intu/nas2_kwater/Videos_web')
    ap.add_argument('--cache-dir', default='/home/intu/projects/pipe-inspector-staging/gpu-server/frame_cache')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    base_projects_dir = Path(args.base_projects_dir)
    videos_web_dir = Path(args.videos_web_dir)
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    print('=' * 80)
    print('[FRAME CACHE] start:', datetime.now().isoformat(timespec='seconds'))
    print('[FRAME CACHE] base_projects_dir:', base_projects_dir)
    print('[FRAME CACHE] videos_web_dir:', videos_web_dir)
    print('[FRAME CACHE] cache_root:', cache_root)

    tasks = collect_tasks(base_projects_dir, videos_web_dir)
    total_videos = len(tasks)
    total_frames = sum(len(v) for v in tasks.values())
    print(f'[FRAME CACHE] target videos={total_videos}, annotated_frames={total_frames}')

    total_known = 0
    total_created = 0
    total_failed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(warm_one_video, vp, frames, cache_root) for vp, frames in tasks.items()]
        for i, fut in enumerate(as_completed(futs), 1):
            known, created, failed = fut.result()
            total_known += known
            total_created += created
            total_failed += failed
            if i % 20 == 0:
                print(f'[FRAME CACHE] progress videos={i}/{total_videos}, created={total_created}, failed={total_failed}')

    print('-' * 80)
    print(f'[FRAME CACHE] done at {datetime.now().isoformat(timespec="seconds")}')
    print(f'[FRAME CACHE] total_known={total_known}, created={total_created}, failed={total_failed}')


if __name__ == '__main__':
    main()
