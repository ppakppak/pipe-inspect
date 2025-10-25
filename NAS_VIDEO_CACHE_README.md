# NAS 비디오 캐시 시스템 가이드

## 개요

2000개 이상의 NAS 비디오를 빠르게 처리하기 위한 메타데이터 캐싱 시스템입니다.

### 주요 기능

- ✅ **메타데이터 사전 추출**: 비디오 정보를 SQLite DB에 캐싱
- ✅ **썸네일 사전 생성**: 320px 썸네일을 미리 생성하여 디스크에 저장
- ✅ **증분 업데이트**: 변경된 파일만 재처리 (mtime 체크)
- ✅ **페이지네이션**: 50개씩 나눠서 로드
- ✅ **디렉토리 메타데이터 파싱**: 하이픈으로 구분된 폴더명 정보 보존

## 사용 방법

### 1단계: 초기 전처리 (최초 1회)

모든 NAS 비디오를 스캔하고 메타데이터 및 썸네일을 생성합니다.

```bash
cd /home/intu/projects/pipe-inspector-electron

# Python 가상환경 활성화
source .venv/bin/activate

# 초기 전처리 실행 (모든 비디오 처리)
python3 preprocess_nas_videos.py

# 또는 기존 캐시 삭제 후 전체 재처리
python3 preprocess_nas_videos.py --clear
```

**예상 소요 시간**: 660개 비디오 기준 약 10-20분

### 2단계: 증분 업데이트 (정기적으로)

새로운 비디오가 추가되거나 기존 비디오가 변경된 경우:

```bash
# 변경된 파일만 처리
python3 preprocess_nas_videos.py --incremental
```

**예상 소요 시간**: 새 비디오 개수에 따라 다름 (변경 없으면 즉시 완료)

### 3단계: 서버 재시작

```bash
./restart.sh
```

## 캐시 디렉토리 구조

```
.video_cache/
├── video_metadata.db       # SQLite 데이터베이스 (메타데이터)
└── thumbnails/             # 썸네일 이미지 (JPEG)
    ├── a1b2c3d4.jpg
    ├── e5f6g7h8.jpg
    └── ...
```

### 캐시 데이터베이스 스키마

```sql
CREATE TABLE video_metadata (
    path TEXT PRIMARY KEY,              -- 비디오 파일 절대 경로
    name TEXT NOT NULL,                 -- 파일명
    nas_folder TEXT NOT NULL,           -- SAHARA 또는 관내시경영상
    parent_dir TEXT NOT NULL,           -- 부모 디렉토리명
    dir_metadata_raw TEXT,              -- 원본 디렉토리명
    dir_metadata_parts TEXT,            -- 파싱된 부분들 (JSON)
    size INTEGER,                       -- 파일 크기 (bytes)
    total_frames INTEGER,               -- 총 프레임 수
    fps REAL,                           -- FPS
    width INTEGER,                      -- 비디오 너비
    height INTEGER,                     -- 비디오 높이
    duration REAL,                      -- 재생시간 (초)
    mtime REAL NOT NULL,                -- 파일 수정 시간
    thumbnail_path TEXT,                -- 썸네일 파일 경로
    cached_at REAL NOT NULL             -- 캐시 생성 시간
);
```

## API 변경 사항

### `/api/nas-videos/list`

**변경 전**: 모든 비디오를 실시간으로 처리하여 반환
**변경 후**: 캐시에서 페이지네이션으로 반환

#### 쿼리 파라미터

- `page` (int): 페이지 번호 (기본값: 1)
- `page_size` (int): 페이지당 개수 (기본값: 50)
- `folder` (string): 폴더 필터 (SAHARA, 관내시경영상)

#### 응답 형식

```json
{
  "success": true,
  "videos": [...],
  "pagination": {
    "page": 1,
    "page_size": 50,
    "total_count": 660,
    "total_pages": 14,
    "has_next": true,
    "has_prev": false
  },
  "cache_stats": {
    "total_videos": 660,
    "by_folder": {
      "SAHARA": 80,
      "관내시경영상": 580
    },
    "thumbnails_generated": 660
  }
}
```

### `/api/nas-videos/thumbnail`

**변경 전**: 요청 시마다 실시간으로 썸네일 생성
**변경 후**: 사전 생성된 썸네일 파일 반환

- 캐시된 썸네일이 있으면 파일 전송
- 없으면 SVG 플레이스홀더 반환

## Frontend 변경 사항

### "더 보기" 버튼

- 처음 50개 비디오 표시
- "더 보기" 버튼 클릭 시 다음 50개 로드
- 모든 비디오 로드 시 버튼 자동 숨김

### 폴더 필터

- 폴더 선택 시 자동으로 페이지 리셋하고 재로드

## 성능 비교

### 변경 전
- 660개 비디오 로드: **30-60초**
- 매 요청마다 OpenCV로 비디오 메타데이터 추출
- 썸네일 실시간 생성

### 변경 후
- 첫 50개 로드: **< 1초**
- 데이터베이스에서 캐시된 메타데이터 조회
- 사전 생성된 썸네일 파일 전송

**성능 향상**: 약 **30-60배** 빠름!

## 유지보수

### 캐시 통계 확인

Python 인터프리터에서:

```python
from video_cache_manager import VideoCacheManager

cache = VideoCacheManager('.video_cache')
stats = cache.get_cache_stats()
print(stats)
```

### 캐시 초기화

```bash
python3 preprocess_nas_videos.py --clear
```

또는 직접 삭제:

```bash
rm -rf .video_cache
```

### 특정 비디오 재처리

Python에서:

```python
from video_cache_manager import VideoCacheManager

cache = VideoCacheManager('.video_cache')
video_path = '/home/intu/nas2_kwater/Videos/SAHARA/video.mp4'
result = cache.process_video(video_path, 'SAHARA')
```

## Cron 작업 (자동 업데이트)

매일 새벽 4시에 증분 업데이트 실행:

```bash
crontab -e

# 다음 줄 추가:
0 4 * * * cd /home/intu/projects/pipe-inspector-electron && /home/intu/projects/pipe-inspector-electron/.venv/bin/python3 preprocess_nas_videos.py --incremental >> /var/log/nas_video_cache.log 2>&1
```

## 문제 해결

### "캐시가 비어있습니다"

전처리가 실행되지 않았을 수 있습니다:

```bash
python3 preprocess_nas_videos.py
```

### "썸네일이 표시되지 않습니다"

캐시 디렉토리 권한 확인:

```bash
ls -la .video_cache/thumbnails/
```

### "메타데이터가 오래되었습니다"

증분 업데이트 실행:

```bash
python3 preprocess_nas_videos.py --incremental
```

## 추가 개선 사항 (향후)

- [ ] 백그라운드 워커로 실시간 캐시 업데이트
- [ ] 여러 해상도의 썸네일 생성 (320px, 640px, 1280px)
- [ ] 비디오 태그/라벨 시스템
- [ ] 검색 기능 (파일명, 디렉토리명)
- [ ] 통계 대시보드
