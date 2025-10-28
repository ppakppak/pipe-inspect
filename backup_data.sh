#!/bin/bash

# 자동 백업 스크립트
# 매일 밤 12시에 실행되어 중요 데이터를 백업합니다

set -e

# 설정
PROJECT_DIR="/home/intu/projects/pipe-inspector-electron"
BACKUP_BASE_DIR="/home/intu/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_BASE_DIR/pipe-inspector_$DATE"
LOG_FILE="$BACKUP_BASE_DIR/backup.log"

# 백업 보관 기간 (일 단위, 30일 이상 된 백업은 삭제)
RETENTION_DAYS=30

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 백업 디렉토리 생성
mkdir -p "$BACKUP_DIR"

log "=========================================="
log "백업 시작"
log "=========================================="

# 1. 프로젝트 데이터 백업 (어노테이션, 토론 등)
log "프로젝트 데이터 백업 중..."
if [ -d "$PROJECT_DIR/projects" ]; then
    tar -czf "$BACKUP_DIR/projects.tar.gz" -C "$PROJECT_DIR" projects 2>/dev/null
    log "✓ 프로젝트 데이터 백업 완료 ($(du -h "$BACKUP_DIR/projects.tar.gz" | cut -f1))"
else
    log "⚠ 프로젝트 디렉토리가 없습니다"
fi

# 2. 사용자 데이터 백업
log "사용자 데이터 백업 중..."
if [ -f "$PROJECT_DIR/users.json" ]; then
    cp "$PROJECT_DIR/users.json" "$BACKUP_DIR/users.json"
    log "✓ 사용자 데이터 백업 완료"
else
    log "⚠ 사용자 파일이 없습니다"
fi

# 3. 비디오 변환 DB 백업
log "비디오 변환 DB 백업 중..."
if [ -d "$PROJECT_DIR/.video_conversion" ]; then
    cp -r "$PROJECT_DIR/.video_conversion" "$BACKUP_DIR/.video_conversion"
    log "✓ 비디오 변환 DB 백업 완료"
fi

# 4. 설정 파일 백업
log "설정 파일 백업 중..."
for file in backend_proxy.py index.html users.json; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        cp "$PROJECT_DIR/$file" "$BACKUP_DIR/"
        log "✓ $file 백업 완료"
    fi
done

# 5. 백업 크기 확인
BACKUP_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log "=========================================="
log "백업 완료: $BACKUP_DIR"
log "백업 크기: $BACKUP_SIZE"
log "=========================================="

# 6. 오래된 백업 삭제 (RETENTION_DAYS 이상)
log "오래된 백업 정리 중..."
find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "pipe-inspector_*" -mtime +$RETENTION_DAYS -exec rm -rf {} \; 2>/dev/null || true
REMAINING_BACKUPS=$(find "$BACKUP_BASE_DIR" -maxdepth 1 -type d -name "pipe-inspector_*" | wc -l)
log "현재 보관 중인 백업: $REMAINING_BACKUPS 개"

# 7. 백업 공간 확인
BACKUP_TOTAL_SIZE=$(du -sh "$BACKUP_BASE_DIR" 2>/dev/null | cut -f1 || echo "N/A")
log "전체 백업 사용 공간: $BACKUP_TOTAL_SIZE"
log "=========================================="

exit 0
