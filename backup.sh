#!/bin/bash
#
# Pipe Inspector 자동 백업 스크립트
# 협업 프로젝트 데이터 안전 보관
#

set -e

# === 설정 ===
PROJECT_DIR="/home/intu/projects/pipe-inspector-electron"
DATA_DIR="/home/intu/Nas2/k_water/pipe_inspector_data"
BACKUP_DIR="/home/intu/backups/pipe-inspector"
LOG_FILE="$BACKUP_DIR/backup.log"

# 보관 기간 (일)
KEEP_DAYS=30

# 날짜 형식
DATE=$(date +%Y%m%d_%H%M%S)
DATE_SHORT=$(date +%Y-%m-%d)

# === 함수 ===
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# === 초기화 ===
mkdir -p "$BACKUP_DIR"/{daily,weekly,code}

log "=========================================="
log "백업 시작: $DATE_SHORT"
log "=========================================="

# === 1. 어노테이션 데이터 백업 (가장 중요) ===
log "[1/5] 어노테이션 데이터 백업 중..."
ANNOTATION_BACKUP="$BACKUP_DIR/daily/annotations_$DATE.tar.gz"

if [ -d "$DATA_DIR" ]; then
    cd "$DATA_DIR"
    find . -name "*.json" -type f | tar -czf "$ANNOTATION_BACKUP" -T -
    
    SIZE=$(du -h "$ANNOTATION_BACKUP" | cut -f1)
    COUNT=$(tar -tzf "$ANNOTATION_BACKUP" | wc -l)
    log "  ✓ 완료: $ANNOTATION_BACKUP ($SIZE, $COUNT files)"
else
    log "  ✗ 데이터 디렉토리 없음: $DATA_DIR"
fi

# === 2. 소스 코드 백업 (전체) ===
log "[2/5] 소스 코드 백업 중..."
CODE_BACKUP="$BACKUP_DIR/daily/source_code_$DATE.tar.gz"

cd "$PROJECT_DIR"
tar -czf "$CODE_BACKUP" \
    --exclude='node_modules' \
    --exclude='.venv' \
    --exclude='.python311_mamba' \
    --exclude='micromamba' \
    --exclude='*.log' \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.video_cache' \
    --exclude='.video_conversion' \
    --exclude='test_output' \
    --exclude='.models' \
    --exclude='inference_results' \
    --exclude='pipe_dataset*' \
    . 2>/dev/null || true

SIZE=$(du -h "$CODE_BACKUP" | cut -f1)
log "  ✓ 완료: $CODE_BACKUP ($SIZE)"

# === 3. GPU 서버 코드 백업 ===
log "[3/5] GPU 서버 코드 백업 중..."
GPU_BACKUP="$BACKUP_DIR/daily/gpu_server_$DATE.tar.gz"

cd "$PROJECT_DIR/gpu-server"
tar -czf "$GPU_BACKUP" \
    --exclude='__pycache__' \
    --exclude='*.log' \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='pipe_dataset*' \
    --exclude='projects' \
    *.py *.json 2>/dev/null || true

SIZE=$(du -h "$GPU_BACKUP" | cut -f1)
log "  ✓ 완료: $GPU_BACKUP ($SIZE)"

# === 4. 사용자 데이터 백업 ===
log "[4/5] 사용자 데이터 백업 중..."
USERS_BACKUP="$BACKUP_DIR/daily/users_$DATE.tar.gz"

cd "$PROJECT_DIR"
tar -czf "$USERS_BACKUP" users.json gpu-server/users.json 2>/dev/null || true
SIZE=$(du -h "$USERS_BACKUP" | cut -f1)
log "  ✓ 완료: $USERS_BACKUP ($SIZE)"

# === 5. 주간 전체 백업 (일요일만) ===
if [ "$(date +%u)" -eq 7 ]; then
    log "[5/5] 주간 전체 백업 중... (일요일)"
    WEEKLY_BACKUP="$BACKUP_DIR/weekly/full_backup_$DATE.tar.gz"
    
    cd "$DATA_DIR"
    tar -czf "$WEEKLY_BACKUP" . 2>/dev/null || true
    
    SIZE=$(du -h "$WEEKLY_BACKUP" | cut -f1)
    log "  ✓ 주간 백업 완료: $WEEKLY_BACKUP ($SIZE)"
else
    log "[5/5] 주간 백업 스킵 (일요일 아님)"
fi

# === 6. 오래된 백업 정리 ===
log "[정리] ${KEEP_DAYS}일 이상 된 백업 삭제 중..."
find "$BACKUP_DIR/daily" -name "*.tar.gz" -mtime +$KEEP_DAYS -delete 2>/dev/null || true
find "$BACKUP_DIR/weekly" -name "*.tar.gz" -mtime +90 -delete 2>/dev/null || true
log "  ✓ 정리 완료"

# === 7. 백업 현황 ===
log ""
log "=== 백업 현황 ==="
TOTAL_SIZE=$(du -sh "$BACKUP_DIR" | cut -f1)
log "총 백업 용량: $TOTAL_SIZE"
log "=========================================="
log "백업 완료!"
log "=========================================="
