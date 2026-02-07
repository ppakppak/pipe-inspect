#!/usr/bin/env python3
"""
코멘트 API를 직접 호출하여 중복 문제 디버깅
"""
import json
from pathlib import Path

BASE_PROJECTS_DIR = Path('./projects')

def scan_all_comments():
    """모든 코멘트가 있는 어노테이션 스캔"""
    all_comments = []

    for user_dir in BASE_PROJECTS_DIR.iterdir():
        if not user_dir.is_dir():
            continue

        user_owner = user_dir.name

        for project_dir in user_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_name = project_dir.name
            project_id = project_name.split('_')[-1] if '_' in project_name else project_name

            annotations_dir = project_dir / 'annotations'
            if not annotations_dir.exists():
                continue

            for video_dir in annotations_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                video_id = video_dir.name

                for json_file in video_dir.glob('*.json'):
                    if json_file.name.endswith('.backup'):
                        continue

                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                    except Exception as e:
                        continue

                    file_owner = json_file.stem

                    if 'annotations' in data:
                        annotations_dict = data['annotations']
                    else:
                        annotations_dict = data

                    for frame_key, frame_annotations in annotations_dict.items():
                        if frame_key in ['project_id', 'video_id', 'user_id', 'user_name', 'updated_at']:
                            continue

                        if not isinstance(frame_annotations, list):
                            continue

                        for ann in frame_annotations:
                            if not isinstance(ann, dict):
                                continue

                            comment = ann.get('comment', '').strip()
                            if comment:
                                all_comments.append({
                                    'project_owner': user_owner,
                                    'project_name': project_name,
                                    'project_id': project_id,
                                    'video_id': video_id,
                                    'frame': frame_key,
                                    'file_owner': file_owner,
                                    'file_path': str(json_file.relative_to(BASE_PROJECTS_DIR)),
                                    'created_by': ann.get('created_by', file_owner),
                                    'label': ann.get('label', 'N/A'),
                                    'comment': comment,
                                    'user_id_in_json': data.get('user_id'),
                                    'user_name_in_json': data.get('user_name')
                                })

    return all_comments

def find_duplicates(comments):
    """같은 프로젝트/비디오/프레임을 가진 중복 코멘트 찾기"""
    from collections import defaultdict

    frame_groups = defaultdict(list)

    for comment in comments:
        key = f"{comment['project_name']}:{comment['video_id']}:{comment['frame']}"
        frame_groups[key].append(comment)

    duplicates = {k: v for k, v in frame_groups.items() if len(v) > 1}
    return duplicates

def main():
    print("=" * 100)
    print("코멘트 중복 디버깅 도구")
    print("=" * 100)
    print()

    print("모든 코멘트 스캔 중...")
    all_comments = scan_all_comments()
    print(f"총 {len(all_comments)}개의 코멘트 발견")
    print()

    print("중복 검사 중...")
    duplicates = find_duplicates(all_comments)

    if not duplicates:
        print("✓ 중복 코멘트를 찾지 못했습니다.")
        print()
        print("모든 코멘트 목록:")
        print("-" * 100)
        for comment in all_comments[:20]:  # 처음 20개만
            print(f"프로젝트: {comment['project_name']}")
            print(f"비디오: {comment['video_id']}")
            print(f"프레임: {comment['frame']}")
            print(f"파일: {comment['file_path']}")
            print(f"파일 소유자: {comment['file_owner']}")
            print(f"created_by: {comment['created_by']}")
            print(f"JSON user_id: {comment['user_id_in_json']}")
            print(f"라벨: {comment['label']}")
            print(f"코멘트: {comment['comment'][:50]}...")
            print()

        if len(all_comments) > 20:
            print(f"... 외 {len(all_comments) - 20}개")

        return

    print(f"❌ {len(duplicates)}개의 중복 프레임 발견!")
    print("=" * 100)
    print()

    for key, comments in duplicates.items():
        project, video, frame = key.split(':')
        print(f"중복 발견: 프로젝트={project}, 비디오={video}, 프레임={frame}")
        print("-" * 100)
        print(f"  → {len(comments)}개의 코멘트:")
        print()

        for i, comment in enumerate(comments, 1):
            print(f"  [{i}] 파일: {comment['file_path']}")
            print(f"      파일 소유자: {comment['file_owner']}")
            print(f"      created_by: {comment['created_by']}")
            print(f"      JSON user_id: {comment['user_id_in_json']}")
            print(f"      JSON user_name: {comment['user_name_in_json']}")
            print(f"      라벨: {comment['label']}")
            print(f"      코멘트: {comment['comment'][:80]}...")
            print()

        print()

    # PE_Test 프레임 961 특별 검색
    print("=" * 100)
    print("PE_Test 프레임 961 검색:")
    print("-" * 100)

    found_pe_test = False
    for comment in all_comments:
        if ('PE' in comment['project_name'].upper() or 'PE' in comment['project_id'].upper()) and '961' in str(comment['frame']):
            found_pe_test = True
            print(f"발견!")
            print(f"  프로젝트: {comment['project_name']}")
            print(f"  프로젝트 ID: {comment['project_id']}")
            print(f"  비디오: {comment['video_id']}")
            print(f"  프레임: {comment['frame']}")
            print(f"  파일: {comment['file_path']}")
            print(f"  파일 소유자: {comment['file_owner']}")
            print(f"  created_by: {comment['created_by']}")
            print(f"  JSON user_id: {comment['user_id_in_json']}")
            print(f"  라벨: {comment['label']}")
            print(f"  코멘트: {comment['comment']}")
            print()

    if not found_pe_test:
        print("PE_Test 프레임 961을 찾지 못했습니다.")
        print("프로젝트명에 'PE'가 포함된 모든 코멘트:")
        for comment in all_comments:
            if 'PE' in comment['project_name'].upper() or 'PE' in comment['project_id'].upper():
                print(f"  - {comment['project_name']}: 프레임 {comment['frame']}, 작성자 {comment['created_by']}")

if __name__ == '__main__':
    main()
