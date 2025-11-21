#!/usr/bin/env python3
"""
비디오 메타데이터 캐시 관리자
NAS 비디오의 메타데이터와 썸네일을 사전 생성하여 캐싱
"""

import sqlite3
import os
import cv2
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class VideoCacheManager:
    """비디오 메타데이터 및 썸네일 캐시 관리"""

    def __init__(self, cache_dir: str = '.video_cache'):
        """
        Args:
            cache_dir: 캐시 디렉토리 경로
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.cache_dir / 'video_metadata.db'
        self.thumbnail_dir = self.cache_dir / 'thumbnails'
        self.thumbnail_dir.mkdir(exist_ok=True)

        self._init_database()

    def _init_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_metadata (
                path TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                nas_folder TEXT NOT NULL,
                parent_dir TEXT NOT NULL,
                dir_metadata_raw TEXT,
                dir_metadata_parts TEXT,
                size INTEGER,
                total_frames INTEGER,
                fps REAL,
                width INTEGER,
                height INTEGER,
                duration REAL,
                mtime REAL NOT NULL,
                thumbnail_path TEXT,
                cached_at REAL NOT NULL
            )
        ''')

        # 인덱스 생성
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_nas_folder ON video_metadata(nas_folder)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_mtime ON video_metadata(mtime)')

        conn.commit()
        conn.close()

    def _get_file_hash(self, file_path: str) -> str:
        """파일 경로의 해시 생성 (썸네일 파일명용)"""
        return hashlib.md5(file_path.encode()).hexdigest()

    def _parse_directory_metadata(self, dir_name: str) -> dict:
        """디렉토리 이름을 파싱하여 메타데이터 추출"""
        # 하이픈으로 분리하고 빈 문자열 제거
        parts = [p.strip() for p in dir_name.split('-') if p.strip()]

        metadata = {
            'raw': dir_name,
            'parts': parts
        }

        # 기본 파트별 저장
        for i, part in enumerate(parts):
            metadata[f'part_{i}'] = part

        # 구조화된 메타데이터 추출 (패턴: 번호-지역-크기-방법)
        if len(parts) >= 4:
            metadata['project_no'] = parts[0]
            metadata['region'] = parts[1]  # 지방/광역
            metadata['pipe_size'] = parts[2]  # 300MM, 500MM 등
            metadata['method'] = parts[3]  # SP, DCIP, HI3P 등
        elif len(parts) >= 3:
            # 일부 디렉토리는 3개 파트만 있을 수 있음
            metadata['region'] = parts[1] if len(parts) > 1 else None
            metadata['pipe_size'] = parts[2] if len(parts) > 2 else None

        return metadata

    def _find_metadata_directory(self, video_path: Path, nas_folder: str) -> str:
        """비디오 파일의 의미 있는 메타데이터 디렉토리를 찾음

        상위 디렉토리를 순회하면서 '번호-지역-크기-방법' 패턴을 가진 디렉토리를 찾음
        예: /SAHARA/329-광역-700MM-SP/2/VTS_01_4.VOB -> '329-광역-700MM-SP'
        """
        import re
        # 번호-지역-크기-방법 패턴 (예: 329-광역-700MM-SP, 13-지방-300MM-DCIP)
        metadata_pattern = re.compile(r'^\d+-[가-힣]+-\d+MM-[A-Z0-9]+$')

        # 비디오 파일의 상위 디렉토리들을 순회
        current = video_path.parent

        # NAS 폴더(SAHARA, 관내시경영상 등)까지만 탐색
        while current.name and current.name != nas_folder:
            dir_name = current.name

            # 패턴 매칭 확인
            if metadata_pattern.match(dir_name):
                return dir_name

            # 하이픈이 3개 이상 포함된 디렉토리도 메타데이터일 가능성이 높음
            if dir_name.count('-') >= 3:
                parts = dir_name.split('-')
                # 첫 번째 파트가 숫자이고, MM이 포함되어 있으면 메타데이터 디렉토리로 간주
                if parts[0].isdigit() and any('MM' in p for p in parts):
                    return dir_name

            current = current.parent

        # 패턴을 찾지 못한 경우 직접 부모 디렉토리 반환
        return video_path.parent.name

    def _generate_thumbnail(self, video_path: str, thumbnail_path: str, frame_number: int = 100) -> bool:
        """비디오 썸네일 생성"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 프레임 번호 조정
            frame_number = min(frame_number, total_frames - 1) if total_frames > 0 else 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                return False

            # 320px 너비로 리사이즈
            height, width = frame.shape[:2]
            new_width = 320
            new_height = int(height * (new_width / width))
            thumbnail = cv2.resize(frame, (new_width, new_height))

            # JPEG로 저장
            cv2.imwrite(thumbnail_path, thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return True

        except Exception as e:
            logger.error(f"썸네일 생성 실패 {video_path}: {e}")
            return False

    def process_video(self, video_path: str, nas_folder: str) -> Optional[Dict]:
        """비디오 메타데이터 추출 및 캐싱"""
        video_path_obj = Path(video_path)

        if not video_path_obj.exists():
            return None

        # 파일 수정 시간
        mtime = video_path_obj.stat().st_mtime

        # 캐시 확인
        cached_data = self.get_cached_metadata(video_path)
        if cached_data and cached_data['mtime'] == mtime:
            # 캐시가 최신이면 반환
            return cached_data

        logger.info(f"비디오 처리 중: {video_path_obj.name}")

        try:
            # 비디오 메타데이터 추출
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()

            # 의미 있는 메타데이터 디렉토리 찾기 (번호-지역-크기-방법 패턴)
            parent_dir = self._find_metadata_directory(video_path_obj, nas_folder)
            dir_metadata = self._parse_directory_metadata(parent_dir)

            # 썸네일 생성
            file_hash = self._get_file_hash(video_path)
            thumbnail_filename = f"{file_hash}.jpg"
            thumbnail_path = str(self.thumbnail_dir / thumbnail_filename)

            self._generate_thumbnail(video_path, thumbnail_path)

            # 메타데이터 구성
            metadata = {
                'path': str(video_path),
                'name': video_path_obj.name,
                'nas_folder': nas_folder,
                'parent_dir': parent_dir,
                'dir_metadata': dir_metadata,
                'size': video_path_obj.stat().st_size,
                'size_mb': round(video_path_obj.stat().st_size / (1024 * 1024), 2),
                'total_frames': total_frames,
                'fps': round(fps, 2),
                'width': width,
                'height': height,
                'duration': round(duration, 2),
                'duration_str': f"{int(duration // 60)}:{int(duration % 60):02d}" if duration > 0 else "0:00",
                'mtime': mtime,
                'thumbnail_path': thumbnail_path,
                'cached_at': datetime.now().timestamp()
            }

            # DB에 저장
            self._save_to_database(metadata)

            return metadata

        except Exception as e:
            logger.error(f"비디오 처리 실패 {video_path}: {e}")
            return None

    def _save_to_database(self, metadata: Dict):
        """메타데이터를 데이터베이스에 저장"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        import json

        cursor.execute('''
            INSERT OR REPLACE INTO video_metadata
            (path, name, nas_folder, parent_dir, dir_metadata_raw, dir_metadata_parts,
             size, total_frames, fps, width, height, duration, mtime, thumbnail_path, cached_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metadata['path'],
            metadata['name'],
            metadata['nas_folder'],
            metadata['parent_dir'],
            metadata['dir_metadata']['raw'],
            json.dumps(metadata['dir_metadata']['parts']),
            metadata['size'],
            metadata['total_frames'],
            metadata['fps'],
            metadata['width'],
            metadata['height'],
            metadata['duration'],
            metadata['mtime'],
            metadata['thumbnail_path'],
            metadata['cached_at']
        ))

        conn.commit()
        conn.close()

    def get_cached_metadata(self, video_path: str) -> Optional[Dict]:
        """캐시된 메타데이터 조회"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM video_metadata WHERE path = ?', (video_path,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        import json

        metadata = dict(row)

        # dir_metadata 재구성
        metadata['dir_metadata'] = {
            'raw': metadata['dir_metadata_raw'],
            'parts': json.loads(metadata['dir_metadata_parts'])
        }

        # 추가 계산 필드
        metadata['size_mb'] = round(metadata['size'] / (1024 * 1024), 2)
        metadata['duration_str'] = f"{int(metadata['duration'] // 60)}:{int(metadata['duration'] % 60):02d}" if metadata['duration'] > 0 else "0:00"

        return metadata

    def get_all_cached_videos(self, nas_folder: Optional[str] = None, limit: int = None, offset: int = 0,
                              region: Optional[str] = None, pipe_size: Optional[str] = None,
                              method: Optional[str] = None) -> List[Dict]:
        """캐시된 모든 비디오 조회 (페이지네이션 및 필터링 지원)

        Args:
            nas_folder: NAS 폴더 필터
            limit: 최대 결과 수
            offset: 시작 오프셋
            region: 지역 필터 (예: '지방', '광역')
            pipe_size: 파이프 크기 필터 (예: '300MM', '500MM')
            method: 방법 필터 (예: 'SP', 'DCIP', 'HI3P')
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 필터 조건이 있으면 모든 레코드를 가져와서 Python에서 필터링
        # (메타데이터가 JSON으로 저장되어 있어 SQL로 필터링하기 어려움)
        if nas_folder:
            query = 'SELECT * FROM video_metadata WHERE nas_folder = ? ORDER BY path'
            cursor.execute(query, (nas_folder,))
        else:
            query = 'SELECT * FROM video_metadata ORDER BY path'
            cursor.execute(query)

        rows = cursor.fetchall()
        conn.close()

        import json

        videos = []
        for row in rows:
            metadata = dict(row)

            # dir_metadata 재구성
            parts = json.loads(metadata['dir_metadata_parts'])
            dir_metadata = self._parse_directory_metadata(metadata['dir_metadata_raw'])
            metadata['dir_metadata'] = dir_metadata

            # 필터 적용
            if region and dir_metadata.get('region') != region:
                continue
            if pipe_size and dir_metadata.get('pipe_size') != pipe_size:
                continue
            if method and dir_metadata.get('method') != method:
                continue

            # 추가 필드 계산
            metadata['size_mb'] = round(metadata['size'] / (1024 * 1024), 2)
            metadata['duration_str'] = f"{int(metadata['duration'] // 60)}:{int(metadata['duration'] % 60):02d}" if metadata['duration'] > 0 else "0:00"

            # 썸네일 존재 여부는 일단 있다고 가정 (필요시 나중에 체크)
            # if metadata['thumbnail_path'] and not os.path.exists(metadata['thumbnail_path']):
            #     metadata['thumbnail_path'] = None

            # 웹 호환 버전 경로 계산 (실제 존재 여부는 나중에 체크, 속도 최적화)
            video_path = metadata['path']
            if '/Videos/' in video_path:
                web_path = video_path.replace('/Videos/', '/Videos_web/').rsplit('.', 1)[0] + '.mp4'
                metadata['web_path'] = web_path
                metadata['has_web_version'] = False  # 속도를 위해 일단 False로 설정
            else:
                metadata['has_web_version'] = False

            videos.append(metadata)

        # 페이지네이션 적용 (필터링 후)
        total_count = len(videos)
        if limit:
            videos = videos[offset:offset + limit]
        elif offset > 0:
            videos = videos[offset:]

        return videos

    def get_cache_stats(self) -> Dict:
        """캐시 통계"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute('SELECT COUNT(*) FROM video_metadata')
        total_count = cursor.fetchone()[0]

        cursor.execute('SELECT nas_folder, COUNT(*) FROM video_metadata GROUP BY nas_folder')
        folder_counts = dict(cursor.fetchall())

        cursor.execute('SELECT COUNT(*) FROM video_metadata WHERE thumbnail_path IS NOT NULL')
        thumbnail_count = cursor.fetchone()[0]

        conn.close()

        return {
            'total_videos': total_count,
            'by_folder': folder_counts,
            'thumbnails_generated': thumbnail_count
        }

    def clear_cache(self):
        """캐시 초기화"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute('DELETE FROM video_metadata')
        conn.commit()
        conn.close()

        # 썸네일 삭제
        for thumbnail_file in self.thumbnail_dir.glob('*.jpg'):
            thumbnail_file.unlink()

        logger.info("캐시 초기화 완료")
