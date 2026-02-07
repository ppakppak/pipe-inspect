#!/usr/bin/env python3
"""
특정 클래스만 필터링하여 YOLO 세그멘테이션 데이터셋 생성 스크립트

사용법:
    python3 build_filtered_dataset.py --project-dir /path/to/project \
                                       --classes slime 소실점 \
                                       --output-dir ./dataset_slime_vanishing \
                                       --split 0.7,0.15,0.15
"""

import argparse
import json
import shutil
import random
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='특정 클래스만 필터링하여 YOLO 데이터셋 생성')
    parser.add_argument('--project-dir', type=str, required=True,
                        help='프로젝트 디렉토리 경로')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='추출할 클래스 목록 (예: slime 소실점)')
    parser.add_argument('--output-dir', type=str, default='filtered_dataset',
                        help='출력 디렉토리 경로')
    parser.add_argument('--split', type=str, default='0.7,0.15,0.15',
                        help='Train/Val/Test 비율 (기본: 0.7,0.15,0.15)')
    parser.add_argument('--videos-web-dir', type=str,
                        default='/home/intu/nas2_kwater/Videos_web',
                        help='웹 호환 비디오 디렉토리')
    return parser.parse_args()


def load_project_config(project_dir):
    """프로젝트 설정 로드"""
    project_file = project_dir / 'project.json'
    if not project_file.exists():
        raise FileNotFoundError(f"project.json not found in {project_dir}")

    with open(project_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_web_video_path(original_path, videos_web_dir):
    """웹 호환 비디오 경로 찾기"""
    original_path = Path(original_path)

    # SAHARA 폴더인 경우
    if 'SAHARA' in str(original_path):
        parts = list(original_path.parts)
        sahara_idx = parts.index('SAHARA')
        relative_path = Path(*parts[sahara_idx+1:])
        web_path = Path(videos_web_dir) / 'SAHARA' / relative_path
        web_path = web_path.with_suffix('.mp4')
    # 관내시경영상 폴더인 경우
    elif '관내시경영상' in str(original_path):
        parts = list(original_path.parts)
        kwan_idx = parts.index('관내시경영상')
        relative_path = Path(*parts[kwan_idx+1:])
        web_path = Path(videos_web_dir) / '관내시경영상' / relative_path
        web_path = web_path.with_suffix('.mp4')
    else:
        # 그대로 사용
        web_path = original_path

    return web_path


def extract_frame(video_path, frame_number):
    """비디오에서 특정 프레임 추출"""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None
    return frame


def polygon_to_yolo(points, img_width, img_height):
    """폴리곤 좌표를 YOLO 세그멘테이션 형식으로 변환"""
    normalized_points = []
    for point in points:
        x = point['x'] / img_width
        y = point['y'] / img_height
        # 0-1 범위로 클리핑
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        normalized_points.append(f"{x:.6f} {y:.6f}")

    return ' '.join(normalized_points)


def collect_filtered_annotations(project_dir, target_classes, videos_web_dir):
    """필터링된 어노테이션 수집"""
    print(f"\n{'='*80}")
    print(f"🔍 어노테이션 수집 중...")
    print(f"{'='*80}")
    print(f"타겟 클래스: {', '.join(target_classes)}")

    project_config = load_project_config(project_dir)
    annotations_dir = project_dir / 'annotations'

    # 클래스 이름 매핑 (한글/영문 모두 지원)
    class_aliases = {
        'slime': ['slime', 'slime(물때)', '슬라임(물때)', '슬라임'],
        '소실점': ['소실점', 'vanishing_point']
    }

    # 타겟 클래스를 표준화
    target_class_set = set()
    for tc in target_classes:
        if tc in class_aliases:
            target_class_set.update(class_aliases[tc])
        else:
            target_class_set.add(tc)

    # 클래스 ID 매핑 생성
    class_to_id = {cls: idx for idx, cls in enumerate(target_classes)}

    collected_frames = []
    stats = {cls: 0 for cls in target_classes}

    # 비디오별 경로 매핑
    video_path_map = {}
    for video in project_config.get('videos', []):
        video_id = video['video_id']
        original_path = video['video_path']
        web_path = find_web_video_path(original_path, videos_web_dir)
        video_path_map[video_id] = {
            'web_path': web_path,
            'width': video.get('width', 1920),
            'height': video.get('height', 1080)
        }

    # 어노테이션 수집
    for video_folder in sorted(annotations_dir.iterdir()):
        if not video_folder.is_dir():
            continue

        video_id = video_folder.name

        if video_id not in video_path_map:
            print(f"⚠️  비디오 정보 없음: {video_id}")
            continue

        video_info = video_path_map[video_id]

        # 비디오 파일 존재 확인
        if not video_info['web_path'].exists():
            print(f"⚠️  비디오 파일 없음: {video_info['web_path']}")
            continue

        # 각 사용자의 어노테이션 파일 읽기
        for json_file in video_folder.glob('*.json'):
            if 'backup' in json_file.name or 'before_fix' in json_file.name:
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                annotations = data.get('annotations', {})

                for frame_num, frame_annos in annotations.items():
                    # 이 프레임에 타겟 클래스가 있는지 확인
                    filtered_annos = []

                    for anno in frame_annos:
                        label = anno.get('label', '')

                        if label in target_class_set:
                            # 원래 클래스로 매핑 (예: 'slime(물때)' -> 'slime')
                            mapped_class = None
                            for original_cls, aliases in class_aliases.items():
                                if label in aliases:
                                    mapped_class = original_cls
                                    break

                            if mapped_class and mapped_class in class_to_id:
                                filtered_annos.append({
                                    'class': mapped_class,
                                    'class_id': class_to_id[mapped_class],
                                    'points': anno.get('polygon', [])
                                })
                                stats[mapped_class] += 1

                    # 타겟 클래스가 있는 프레임만 수집
                    if filtered_annos:
                        collected_frames.append({
                            'video_id': video_id,
                            'video_path': video_info['web_path'],
                            'frame_number': int(frame_num),
                            'width': video_info['width'],
                            'height': video_info['height'],
                            'annotations': filtered_annos,
                            'user': json_file.stem
                        })

            except Exception as e:
                print(f"❌ 오류 ({json_file.name}): {e}")

    print(f"\n✅ 수집 완료:")
    print(f"   - 총 프레임: {len(collected_frames)}개")
    for cls, count in stats.items():
        print(f"   - {cls}: {count}개")

    return collected_frames, class_to_id


def split_dataset(frames, split_ratio):
    """데이터셋을 train/val/test로 분할"""
    train_ratio, val_ratio, test_ratio = map(float, split_ratio.split(','))
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total

    # 셔플
    random.shuffle(frames)

    total_frames = len(frames)
    train_end = int(total_frames * train_ratio)
    val_end = train_end + int(total_frames * val_ratio)

    return {
        'train': frames[:train_end],
        'val': frames[train_end:val_end],
        'test': frames[val_end:]
    }


def build_dataset(frames_dict, class_mapping, output_dir):
    """YOLO 데이터셋 빌드"""
    output_path = Path(output_dir)

    # 디렉토리 생성
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"🏗️  데이터셋 빌드 중...")
    print(f"{'='*80}")

    total_saved = 0

    for split, frames in frames_dict.items():
        print(f"\n📦 {split.upper()} 세트 생성 중... ({len(frames)}개 프레임)")

        saved_count = 0
        for idx, frame_data in enumerate(frames):
            video_path = frame_data['video_path']
            frame_num = frame_data['frame_number']
            width = frame_data['width']
            height = frame_data['height']
            annotations = frame_data['annotations']

            # 프레임 추출
            frame_img = extract_frame(video_path, frame_num)
            if frame_img is None:
                print(f"⚠️  프레임 추출 실패: {video_path} #{frame_num}")
                continue

            # 파일명 생성
            video_id = frame_data['video_id']
            filename = f"{video_id}_frame_{frame_num:06d}"

            # 이미지 저장
            img_path = output_path / split / 'images' / f"{filename}.jpg"
            cv2.imwrite(str(img_path), frame_img)

            # 라벨 파일 생성
            label_path = output_path / split / 'labels' / f"{filename}.txt"
            with open(label_path, 'w') as f:
                for anno in annotations:
                    class_id = anno['class_id']
                    points = anno['points']

                    if len(points) < 3:  # 최소 3개 점 필요
                        continue

                    yolo_coords = polygon_to_yolo(points, width, height)
                    f.write(f"{class_id} {yolo_coords}\n")

            saved_count += 1

            if (idx + 1) % 50 == 0:
                print(f"   진행: {idx + 1}/{len(frames)} 프레임")

        print(f"   ✅ {split}: {saved_count}개 저장됨")
        total_saved += saved_count

    # data.yaml 생성
    yaml_content = f"""# YOLO Segmentation Dataset
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {output_path.absolute()}
train: train/images
val: val/images
test: test/images

nc: {len(class_mapping)}
names: {list(class_mapping.keys())}
"""

    yaml_path = output_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n{'='*80}")
    print(f"✅ 데이터셋 빌드 완료!")
    print(f"{'='*80}")
    print(f"총 저장된 이미지: {total_saved}개")
    print(f"출력 디렉토리: {output_path.absolute()}")
    print(f"설정 파일: {yaml_path}")
    print(f"{'='*80}\n")

    return total_saved


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print(f"🎯 YOLO 필터링 데이터셋 빌더")
    print(f"{'='*80}")
    print(f"프로젝트: {args.project_dir}")
    print(f"타겟 클래스: {', '.join(args.classes)}")
    print(f"출력 디렉토리: {args.output_dir}")
    print(f"Split 비율: {args.split}")

    project_dir = Path(args.project_dir)
    if not project_dir.exists():
        print(f"❌ 프로젝트 디렉토리가 존재하지 않습니다: {project_dir}")
        return

    # 1. 어노테이션 수집
    frames, class_mapping = collect_filtered_annotations(
        project_dir,
        args.classes,
        args.videos_web_dir
    )

    if not frames:
        print("\n❌ 타겟 클래스의 어노테이션을 찾을 수 없습니다.")
        return

    # 2. 데이터셋 분할
    print(f"\n{'='*80}")
    print(f"📊 데이터셋 분할 중...")
    print(f"{'='*80}")
    frames_dict = split_dataset(frames, args.split)

    for split, split_frames in frames_dict.items():
        print(f"   {split}: {len(split_frames)}개")

    # 3. 데이터셋 빌드
    total_saved = build_dataset(frames_dict, class_mapping, args.output_dir)

    if total_saved > 0:
        print(f"\n✨ 성공적으로 {total_saved}개의 이미지로 데이터셋을 생성했습니다!")
        print(f"📁 {Path(args.output_dir).absolute()}/data.yaml 을 확인하세요.\n")
    else:
        print(f"\n❌ 데이터셋 생성 실패\n")


if __name__ == '__main__':
    random.seed(42)  # 재현 가능한 결과를 위해
    main()
