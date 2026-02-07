#!/usr/bin/env python3
"""
annotation.json 같은 잘못된 파일명을 찾아서 수정하는 스크립트
"""
import json
from pathlib import Path
import shutil
from datetime import datetime

BASE_PROJECTS_DIR = Path('./projects')

def find_invalid_annotation_files():
    """annotation.json 같은 잘못된 파일명 찾기"""
    invalid_files = []

    for user_dir in BASE_PROJECTS_DIR.iterdir():
        if not user_dir.is_dir():
            continue

        for project_dir in user_dir.iterdir():
            if not project_dir.is_dir():
                continue

            annotations_dir = project_dir / 'annotations'
            if not annotations_dir.exists():
                continue

            for video_dir in annotations_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                for json_file in video_dir.glob('*.json'):
                    # annotation.json 파일 찾기
                    if json_file.stem == 'annotation':
                        invalid_files.append(json_file)

    return invalid_files

def analyze_annotation_file(file_path):
    """어노테이션 파일 분석"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        user_id = data.get('user_id')
        user_name = data.get('user_name')

        # annotations 개수 계산
        annotations_dict = data.get('annotations', data)
        total_annotations = 0
        frames_with_comments = []

        for frame_key, frame_annotations in annotations_dict.items():
            if frame_key in ['project_id', 'video_id', 'user_id', 'user_name', 'updated_at']:
                continue
            if isinstance(frame_annotations, list):
                total_annotations += len(frame_annotations)
                # 코멘트가 있는 어노테이션 찾기
                for ann in frame_annotations:
                    if isinstance(ann, dict) and ann.get('comment'):
                        frames_with_comments.append({
                            'frame': frame_key,
                            'label': ann.get('label'),
                            'comment': ann.get('comment'),
                            'created_by': ann.get('created_by')
                        })

        return {
            'file_path': str(file_path),
            'user_id': user_id,
            'user_name': user_name,
            'total_annotations': total_annotations,
            'frames_with_comments': frames_with_comments
        }
    except Exception as e:
        return {
            'file_path': str(file_path),
            'error': str(e)
        }

def main():
    print("=" * 80)
    print("어노테이션 파일 분석 도구")
    print("=" * 80)
    print()

    invalid_files = find_invalid_annotation_files()

    if not invalid_files:
        print("✓ 'annotation.json' 파일을 찾지 못했습니다.")
        print()
        print("다른 검사를 수행합니다...")
        print()

        # 모든 어노테이션 파일에서 created_by가 파일명과 다른 경우 찾기
        print("파일명과 created_by가 다른 어노테이션 찾기:")
        print("-" * 80)

        mismatched_count = 0
        for user_dir in BASE_PROJECTS_DIR.iterdir():
            if not user_dir.is_dir():
                continue

            for project_dir in user_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                annotations_dir = project_dir / 'annotations'
                if not annotations_dir.exists():
                    continue

                for video_dir in annotations_dir.iterdir():
                    if not video_dir.is_dir():
                        continue

                    for json_file in video_dir.glob('*.json'):
                        if json_file.stem.endswith('.backup'):
                            continue

                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)

                            file_owner = json_file.stem
                            annotations_dict = data.get('annotations', data)

                            for frame_key, frame_annotations in annotations_dict.items():
                                if frame_key in ['project_id', 'video_id', 'user_id', 'user_name', 'updated_at']:
                                    continue
                                if isinstance(frame_annotations, list):
                                    for ann in frame_annotations:
                                        if isinstance(ann, dict):
                                            created_by = ann.get('created_by')
                                            if created_by and created_by != file_owner:
                                                mismatched_count += 1
                                                print(f"파일: {json_file.relative_to(BASE_PROJECTS_DIR)}")
                                                print(f"  파일 소유자: {file_owner}")
                                                print(f"  created_by: {created_by}")
                                                print(f"  프레임: {frame_key}")
                                                if ann.get('comment'):
                                                    print(f"  코멘트: {ann.get('comment')[:50]}...")
                                                print()
                        except Exception as e:
                            pass

        if mismatched_count == 0:
            print("✓ 불일치하는 어노테이션을 찾지 못했습니다.")
        else:
            print(f"총 {mismatched_count}개의 불일치하는 어노테이션을 찾았습니다.")

        return

    print(f"발견된 'annotation.json' 파일: {len(invalid_files)}개")
    print("=" * 80)
    print()

    for file_path in invalid_files:
        print(f"파일: {file_path.relative_to(BASE_PROJECTS_DIR)}")
        print("-" * 80)

        analysis = analyze_annotation_file(file_path)

        if 'error' in analysis:
            print(f"❌ 오류: {analysis['error']}")
        else:
            print(f"사용자 ID: {analysis['user_id']}")
            print(f"사용자 이름: {analysis['user_name']}")
            print(f"총 어노테이션 개수: {analysis['total_annotations']}")
            print(f"코멘트가 있는 프레임 개수: {len(analysis['frames_with_comments'])}")

            if analysis['frames_with_comments']:
                print("\n코멘트가 있는 프레임:")
                for fc in analysis['frames_with_comments']:
                    print(f"  - 프레임 {fc['frame']}: {fc['label']} (created_by: {fc.get('created_by', 'N/A')})")
                    print(f"    코멘트: {fc['comment'][:60]}...")

            # 수정 제안
            print("\n제안된 조치:")
            if analysis['user_id'] and analysis['user_id'] != 'annotation':
                new_filename = file_path.parent / f"{analysis['user_id']}.json"
                if new_filename.exists():
                    print(f"  ⚠️  {analysis['user_id']}.json 파일이 이미 존재합니다!")
                    print(f"  → 두 파일을 병합해야 합니다.")
                else:
                    print(f"  ✓ {new_filename.name}으로 이름 변경 가능")
            else:
                print(f"  ⚠️  user_id가 없거나 'annotation'입니다. 수동으로 확인 필요")

        print()
        print()

if __name__ == '__main__':
    main()
