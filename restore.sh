#!/bin/bash
#
# Pipe Inspector 복구 스크립트
# 백업에서 데이터 복구
#

set -e

BACKUP_DIR="/home/intu/backups/pipe-inspector"
DATA_DIR="/home/intu/Nas2/k_water/pipe_inspector_data"

echo "=========================================="
echo "  Pipe Inspector 복구 도구"
echo "=========================================="
echo ""

# 사용 가능한 백업 목록
echo "📦 사용 가능한 백업:"
echo ""
echo "[Daily 백업]"
ls -lht "$BACKUP_DIR/daily/"annotations_*.tar.gz 2>/dev/null | head -10 | nl
echo ""
echo "[Weekly 백업]"
ls -lht "$BACKUP_DIR/weekly/"*.tar.gz 2>/dev/null | head -5 | nl
echo ""

# 복구할 백업 선택
read -p "복구할 백업 파일 경로 입력 (또는 번호): " BACKUP_FILE

# 번호면 파일 경로로 변환
if [[ "$BACKUP_FILE" =~ ^[0-9]+$ ]]; then
    BACKUP_FILE=$(ls -t "$BACKUP_DIR/daily/"annotations_*.tar.gz 2>/dev/null | sed -n "${BACKUP_FILE}p")
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ 백업 파일을 찾을 수 없습니다: $BACKUP_FILE"
    exit 1
fi

echo ""
echo "선택된 백업: $BACKUP_FILE"
echo "복구 대상: $DATA_DIR"
echo ""

# 확인
read -p "⚠️  기존 데이터가 덮어쓰여집니다. 계속하시겠습니까? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    echo "복구 취소됨"
    exit 0
fi

# 현재 데이터 백업
echo ""
echo "📋 현재 데이터 임시 백업 중..."
TEMP_BACKUP="/tmp/pipe_restore_backup_$(date +%Y%m%d_%H%M%S).tar.gz"
cd "$DATA_DIR"
tar -czf "$TEMP_BACKUP" . 2>/dev/null || true
echo "  ✓ 임시 백업: $TEMP_BACKUP"

# 복구 실행
echo ""
echo "🔄 복구 중..."
cd "$DATA_DIR"
tar -xzf "$BACKUP_FILE"

echo ""
echo "=========================================="
echo "✅ 복구 완료!"
echo "=========================================="
echo ""
echo "문제 발생 시 임시 백업에서 복원:"
echo "  cd $DATA_DIR && tar -xzf $TEMP_BACKUP"
