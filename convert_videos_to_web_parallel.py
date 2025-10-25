#!/usr/bin/env python3
"""
NAS 비디오를 웹 브라우저 호환 MP4로 변환 (병렬 처리 + GPU 가속)
원본 폴더 구조를 유지하면서 Videos_web 폴더에 저장
"""

import os
import subprocess
import sqlite3
import json
from pathlib import Path
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 설정
SOURCE_BASE = Path('/home/intu/nas2_kwater/Videos')
TARGET_BASE = Path('/home/intu/nas2_kwater/Videos_web')
VIDEO_EXTENSIONS = {'.avi', '.AVI', '.vob', '.VOB', '.mov', '.MOV', '.mkv', '.MKV'}

# 변환 데이터베이스
CONVERSION_DB = Path('.video_conversion/conversion.db')

# 병렬 처리 설정
MAX_WORKERS = 6  # RTX 3090은 동시에 여러 스트림 인코딩 가능


class VideoConverter:
    """비디오 변환 관리"""

    def __init__(self, use_gpu=True):
        self.db_path = CONVERSION_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu
        self._init_database()

    def _init_database(self):
        """변환 기록 데이터베이스 초기화"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversions (
                source_path TEXT PRIMARY KEY,
                target_path TEXT NOT NULL,
                source_mtime REAL NOT NULL,
                converted_at REAL NOT NULL,
                file_size INTEGER,
                duration REAL,
                status TEXT DEFAULT 'completed'
            )
        ''')

        conn.commit()
        conn.close()

    def is_converted(self, source_path: str) -> tuple[bool, str]:
        """비디오가 이미 변환되었는지 확인

        Returns:
            (변환됨 여부, 변환된 파일 경로)
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT target_path, source_mtime FROM conversions WHERE source_path = ?',
                      (source_path,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return False, ""

        target_path, recorded_mtime = row

        # 원본 파일의 현재 mtime 확인
        source_file = Path(source_path)
        if not source_file.exists():
            return False, ""

        current_mtime = source_file.stat().st_mtime

        # mtime이 변경되었으면 재변환 필요
        if abs(current_mtime - recorded_mtime) > 1:  # 1초 이상 차이나면
            return False, ""

        # 변환된 파일이 실제로 존재하는지 확인
        if not Path(target_path).exists():
            return False, ""

        return True, target_path

    def _check_codec(self, video_path: str) -> dict:
        """비디오 코덱 확인

        Returns:
            {'video_codec': 'h264', 'audio_codec': 'aac', 'needs_conversion': False}
        """
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                return {'needs_conversion': True}

            import json
            data = json.loads(result.stdout)

            video_codec = None
            audio_codec = None

            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_codec = stream.get('codec_name')
                elif stream.get('codec_type') == 'audio':
                    audio_codec = stream.get('codec_name')

            # H264 비디오면 변환 불필요 (오디오는 상관없음)
            if video_codec == 'h264':
                if audio_codec is None:
                    needs_conversion = False
                elif audio_codec == 'aac':
                    needs_conversion = False
                else:
                    # 비디오 H264 + 오디오 다른 코덱 → 오디오만 재인코딩 필요
                    needs_conversion = True
            else:
                # 비디오가 H264가 아님 → 재인코딩 필요
                needs_conversion = True

            return {
                'video_codec': video_codec,
                'audio_codec': audio_codec,
                'needs_conversion': needs_conversion
            }
        except Exception as e:
            logger.warning(f"코덱 확인 실패 {video_path}: {e}")
            return {'needs_conversion': True}

    def convert_video(self, source_path: str, target_path: str) -> bool:
        """FFmpeg로 비디오 변환 (GPU 가속 사용)

        Args:
            source_path: 원본 비디오 경로
            target_path: 변환된 비디오 저장 경로

        Returns:
            성공 여부
        """
        source_file = Path(source_path)
        target_file = Path(target_path)

        # 타겟 디렉토리 생성
        target_file.parent.mkdir(parents=True, exist_ok=True)

        # 코덱 확인
        codec_info = self._check_codec(source_path)

        if not codec_info['needs_conversion']:
            # 이미 H264 (오디오 AAC 또는 없음): 스트림 복사 (매우 빠름)
            audio_info = f"+{codec_info.get('audio_codec')}" if codec_info.get('audio_codec') else "(오디오 없음)"
            logger.info(f"[{mp.current_process().name}] 이미 H264{audio_info}, 복사: {source_file.name}")
            cmd = [
                'ffmpeg',
                '-i', str(source_file),
                '-c', 'copy',
                '-movflags', '+faststart',
                '-y',
                str(target_file)
            ]
        else:
            # 변환 필요
            logger.info(f"[{mp.current_process().name}] 변환 필요 (코덱: {codec_info.get('video_codec')}/{codec_info.get('audio_codec') or 'None'})")

            if self.use_gpu:
                # GPU 가속 인코딩 (h264_nvenc) - 10-50배 빠름!
                # hwaccel 없이 간단한 GPU 인코딩만 사용
                cmd = [
                    'ffmpeg',
                    '-i', str(source_file),
                    '-c:v', 'h264_nvenc',  # NVIDIA GPU 인코더
                    '-preset', 'fast',  # 빠른 인코딩
                    '-crf', '23',  # 품질
                ]
            else:
                # CPU 인코딩 (fallback)
                cmd = [
                    'ffmpeg',
                    '-i', str(source_file),
                    '-c:v', 'libx264',
                    '-preset', 'veryfast',  # CPU에선 더 빠른 preset
                    '-crf', '23'
                ]

            # 오디오가 있으면 AAC로 인코딩, 없으면 오디오 스트림 제거
            if codec_info.get('audio_codec'):
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            else:
                cmd.append('-an')  # 오디오 없음

            # 마지막 옵션들
            cmd.extend(['-movflags', '+faststart', '-y', str(target_file)])

        try:
            logger.info(f"[{mp.current_process().name}] 변환 시작: {source_file.name}")

            # FFmpeg 실행
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600  # 10분 타임아웃
            )

            if result.returncode != 0:
                logger.error(f"[{mp.current_process().name}] 변환 실패: {source_file.name}")
                logger.error(f"FFmpeg error: {result.stderr[-500:]}")  # 마지막 500자만
                return False

            # 변환 성공 - DB에 기록
            mtime = source_file.stat().st_mtime
            file_size = target_file.stat().st_size

            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO conversions
                (source_path, target_path, source_mtime, converted_at, file_size, status)
                VALUES (?, ?, ?, ?, ?, 'completed')
            ''', (
                str(source_path),
                str(target_path),
                mtime,
                datetime.now().timestamp(),
                file_size
            ))
            conn.commit()
            conn.close()

            logger.info(f"[{mp.current_process().name}] ✓ 변환 완료: {source_file.name} ({file_size / (1024*1024):.2f} MB)")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"[{mp.current_process().name}] 변환 타임아웃: {source_file.name}")
            return False
        except Exception as e:
            logger.error(f"[{mp.current_process().name}] 변환 중 오류: {source_file.name} - {e}")
            return False


def process_single_video(args):
    """단일 비디오 처리 (병렬 처리용)"""
    source_file, target_path, incremental, use_gpu = args

    converter = VideoConverter(use_gpu=use_gpu)

    # 이미 변환되었는지 확인
    if incremental:
        is_done, existing_path = converter.is_converted(str(source_file))
        if is_done:
            return 'skipped', str(source_file)

    # 변환 실행
    if converter.convert_video(str(source_file), str(target_path)):
        return 'converted', str(source_file)
    else:
        return 'failed', str(source_file)


def scan_and_convert_parallel(source_folder: str, incremental: bool = True, use_gpu: bool = True, max_workers: int = MAX_WORKERS):
    """폴더를 스캔하여 병렬로 비디오 변환

    Args:
        source_folder: 소스 폴더 (SAHARA 또는 관내시경영상)
        incremental: True면 새 파일만, False면 전체 재변환
        use_gpu: GPU 가속 사용 여부
        max_workers: 병렬 작업 수
    """
    source_base = SOURCE_BASE / source_folder
    target_base = TARGET_BASE / source_folder

    if not source_base.exists():
        logger.error(f"소스 폴더가 존재하지 않습니다: {source_base}")
        return

    logger.info(f"스캔 시작: {source_base}")
    logger.info(f"타겟: {target_base}")
    logger.info(f"병렬 처리: {max_workers}개 프로세스")
    logger.info(f"GPU 가속: {'사용' if use_gpu else '미사용'}")

    # 모든 비디오 파일 찾기
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(source_base.rglob(f'*{ext}'))

    logger.info(f"총 {len(video_files)}개 비디오 파일 발견")

    # 작업 목록 생성
    tasks = []
    for source_file in video_files:
        rel_path = source_file.relative_to(source_base)
        target_path = target_base / rel_path.with_suffix('.mp4')
        tasks.append((source_file, target_path, incremental, use_gpu))

    # 변환 통계
    converted_count = 0
    skipped_count = 0
    failed_count = 0

    # 병렬 처리
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, task): task for task in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            source_file = task[0]

            try:
                status, path = future.result()

                if status == 'converted':
                    converted_count += 1
                elif status == 'skipped':
                    skipped_count += 1
                else:
                    failed_count += 1

                # 진행 상황 출력
                logger.info(f"진행: {i}/{len(tasks)} - 변환:{converted_count} 스킵:{skipped_count} 실패:{failed_count}")

            except Exception as e:
                logger.error(f"작업 처리 중 오류: {source_file} - {e}")
                failed_count += 1

    # 결과 요약
    logger.info("=" * 60)
    logger.info("변환 완료!")
    logger.info(f"  변환됨: {converted_count}")
    logger.info(f"  스킵됨: {skipped_count}")
    logger.info(f"  실패: {failed_count}")
    logger.info("=" * 60)


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='NAS 비디오를 웹 호환 MP4로 병렬 변환 (GPU 가속)')
    parser.add_argument('--folder', choices=['SAHARA', '관내시경영상', 'all'],
                       default='all', help='변환할 폴더')
    parser.add_argument('--full', action='store_true',
                       help='전체 재변환 (기본: 증분 변환)')
    parser.add_argument('--no-gpu', action='store_true',
                       help='GPU 가속 비활성화 (CPU만 사용)')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS,
                       help=f'병렬 작업 수 (기본: {MAX_WORKERS})')

    args = parser.parse_args()

    incremental = not args.full
    use_gpu = not args.no_gpu

    if args.folder == 'all':
        scan_and_convert_parallel('SAHARA', incremental, use_gpu, args.workers)
        scan_and_convert_parallel('관내시경영상', incremental, use_gpu, args.workers)
    else:
        scan_and_convert_parallel(args.folder, incremental, use_gpu, args.workers)


if __name__ == '__main__':
    main()
