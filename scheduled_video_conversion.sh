#!/bin/bash
# 매일 자정(00:00)~오전 9시(09:00) 사이에 비디오 변환 작업 수행
# crontab으로 자정에 시작, 오전 9시 전에 자동 종료

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="$SCRIPT_DIR/scheduled_conversion.log"
PID_FILE="$SCRIPT_DIR/.scheduled_conversion.pid"

# 로그 함수
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 종료 시간 체크 함수 (09시~23시 사이에는 중지)
is_time_to_stop() {
    current_hour=$(date +%H)
    # 9시~23시 사이는 작업 시간이므로 변환 중지
    if [ "$current_hour" -ge 9 ] && [ "$current_hour" -lt 24 ]; then
        return 0  # true: 작업 시간 (변환 중지)
    else
        return 1  # false: 변환 가능 시간 (00~08시)
    fi
}

# 이미 실행 중인지 확인
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        log "변환 작업이 이미 실행 중입니다 (PID: $OLD_PID)"
        exit 0
    fi
fi

log "=========================================="
log "예약된 비디오 변환 작업 시작"
log "=========================================="

# 현재 PID 저장
echo $$ > "$PID_FILE"

# 가상환경 활성화
if [ -d "$SCRIPT_DIR/.venv" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    log "ERROR: 가상환경을 찾을 수 없습니다: $SCRIPT_DIR/.venv"
    rm -f "$PID_FILE"
    exit 1
fi

# Worker 수 설정 (새벽 시간이므로 더 많은 worker 사용 가능)
WORKERS=4

# 종료 트랩 설정
cleanup() {
    log "변환 작업 종료 중..."
    # 실행 중인 변환 프로세스 종료
    pkill -P $$ -f convert_videos_to_web_parallel.py
    rm -f "$PID_FILE"
    log "변환 작업 종료 완료"
    exit 0
}

trap cleanup SIGTERM SIGINT

# 변환 작업 실행 함수
run_conversion() {
    local folder=$1
    log "폴더 변환 시작: $folder (증분 모드, Workers: $WORKERS)"

    # 백그라운드로 변환 실행
    python3 "$SCRIPT_DIR/convert_videos_to_web_parallel.py" \
        --folder "$folder" \
        --workers "$WORKERS" \
        >> "$LOG_FILE" 2>&1 &

    CONVERSION_PID=$!

    # 변환 작업 모니터링 (8시까지만)
    while true; do
        # 프로세스가 종료되었는지 확인
        if ! ps -p "$CONVERSION_PID" > /dev/null 2>&1; then
            log "폴더 '$folder' 변환 완료"
            break
        fi

        # 8시가 되었는지 확인
        if is_time_to_stop; then
            log "9시가 되어 변환 작업을 중지합니다"
            kill "$CONVERSION_PID" 2>/dev/null
            wait "$CONVERSION_PID" 2>/dev/null
            return 1  # 시간 초과
        fi

        # 10초마다 체크
        sleep 10
    done

    return 0  # 정상 완료
}

# SAHARA 폴더 변환
run_conversion "SAHARA"
if [ $? -ne 0 ]; then
    log "시간 초과로 SAHARA 폴더 변환 중단"
    cleanup
fi

# 시간 체크
if is_time_to_stop; then
    log "9시가 되어 변환 작업을 종료합니다"
    cleanup
fi

# 관내시경영상 폴더 변환
run_conversion "관내시경영상"
if [ $? -ne 0 ]; then
    log "시간 초과로 관내시경영상 폴더 변환 중단"
    cleanup
fi

log "=========================================="
log "모든 변환 작업 완료"
log "=========================================="

# 정상 종료
rm -f "$PID_FILE"
exit 0
