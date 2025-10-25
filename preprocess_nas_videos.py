#!/usr/bin/env python3
"""
NAS 비디오 전처리 스크립트
모든 NAS 비디오의 메타데이터와 썸네일을 사전 생성하여 캐싱
"""

import logging
import argparse
import time
from pathlib import Path
from video_cache_manager import VideoCacheManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# NAS 비디오 경로 설정
NAS_VIDEO_PATHS = [
    '/home/intu/nas2_kwater/Videos/SAHARA',
    '/home/intu/nas2_kwater/Videos/관내시경영상'
]


def find_videos(directory: str, min_size_mb: int = 10):
    """디렉토리에서 비디오 파일 찾기"""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.vob'}
    path = Path(directory)

    if not path.exists():
        logger.warning(f"경로가 존재하지 않습니다: {directory}")
        return []

    min_size_bytes = min_size_mb * 1024 * 1024  # MB를 바이트로 변환
    videos = []
    for video_file in path.rglob('*'):
        if video_file.is_file() and video_file.suffix.lower() in video_extensions:
            # 최소 파일 크기 필터 (DVD 메뉴 파일 등 제외)
            if video_file.stat().st_size >= min_size_bytes:
                videos.append(str(video_file))

    return videos


def main():
    parser = argparse.ArgumentParser(description='NAS 비디오 전처리 스크립트')
    parser.add_argument('--cache-dir', default='.video_cache', help='캐시 디렉토리 경로')
    parser.add_argument('--clear', action='store_true', help='기존 캐시 초기화')
    parser.add_argument('--incremental', action='store_true', help='증분 업데이트 (변경된 파일만)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NAS 비디오 전처리 시작")
    logger.info("=" * 60)

    # 캐시 매니저 초기화
    cache_manager = VideoCacheManager(cache_dir=args.cache_dir)

    # 캐시 초기화 옵션
    if args.clear:
        logger.info("기존 캐시 초기화...")
        cache_manager.clear_cache()

    # 기존 캐시 통계
    stats = cache_manager.get_cache_stats()
    logger.info(f"현재 캐시 상태: {stats['total_videos']}개 비디오")

    # 각 NAS 경로에서 비디오 수집
    all_videos = []
    for nas_path in NAS_VIDEO_PATHS:
        logger.info(f"\n📁 스캔 중: {nas_path}")
        videos = find_videos(nas_path)
        logger.info(f"   발견된 비디오: {len(videos)}개")

        folder_name = Path(nas_path).name
        for video_path in videos:
            all_videos.append((video_path, folder_name))

    logger.info(f"\n총 {len(all_videos)}개 비디오 발견")

    # 증분 업데이트 모드
    if args.incremental:
        logger.info("\n증분 업데이트 모드: 변경된 파일만 처리합니다...")
        videos_to_process = []
        for video_path, folder_name in all_videos:
            cached = cache_manager.get_cached_metadata(video_path)
            if not cached:
                videos_to_process.append((video_path, folder_name))
            else:
                # mtime 체크
                current_mtime = Path(video_path).stat().st_mtime
                if cached['mtime'] != current_mtime:
                    videos_to_process.append((video_path, folder_name))

        logger.info(f"처리할 비디오: {len(videos_to_process)}개")
        all_videos = videos_to_process

    # 비디오 처리
    logger.info(f"\n" + "=" * 60)
    logger.info("비디오 처리 시작")
    logger.info("=" * 60)

    start_time = time.time()
    processed_count = 0
    failed_count = 0
    skipped_count = 0

    for idx, (video_path, folder_name) in enumerate(all_videos, 1):
        try:
            logger.info(f"[{idx}/{len(all_videos)}] 처리 중: {Path(video_path).name}")

            result = cache_manager.process_video(video_path, folder_name)

            if result:
                processed_count += 1
            else:
                failed_count += 1

            # 진행률 표시
            if idx % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                remaining = (len(all_videos) - idx) * avg_time
                logger.info(
                    f"   진행률: {idx}/{len(all_videos)} ({idx/len(all_videos)*100:.1f}%) | "
                    f"성공: {processed_count} | 실패: {failed_count} | "
                    f"남은 시간: {int(remaining//60)}분 {int(remaining%60)}초"
                )

        except KeyboardInterrupt:
            logger.info("\n사용자 중단")
            break
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            failed_count += 1

    # 최종 통계
    elapsed_time = time.time() - start_time
    logger.info(f"\n" + "=" * 60)
    logger.info("처리 완료")
    logger.info("=" * 60)
    logger.info(f"총 소요 시간: {int(elapsed_time//60)}분 {int(elapsed_time%60)}초")
    logger.info(f"처리된 비디오: {processed_count}개")
    logger.info(f"실패한 비디오: {failed_count}개")

    # 최종 캐시 통계
    final_stats = cache_manager.get_cache_stats()
    logger.info(f"\n최종 캐시 상태:")
    logger.info(f"  총 비디오: {final_stats['total_videos']}개")
    logger.info(f"  폴더별 통계:")
    for folder, count in final_stats['by_folder'].items():
        logger.info(f"    - {folder}: {count}개")
    logger.info(f"  썸네일 생성: {final_stats['thumbnails_generated']}개")
    logger.info(f"\n캐시 디렉토리: {cache_manager.cache_dir}")


if __name__ == '__main__':
    main()
