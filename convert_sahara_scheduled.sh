#!/bin/bash

# SAHARA 비디오 변환 - 새벽 시간대 전용 (01:00 ~ 09:00)
# 현재 시간 체크 후 실행

CURRENT_HOUR=$(date +%H)

# 01시부터 08시 사이에만 실행 (09시는 제외, 종료 시간 여유)
if [ "$CURRENT_HOUR" -ge 1 ] && [ "$CURRENT_HOUR" -lt 9 ]; then
    echo "[$(date)] SAHARA 변환 작업 시작 (시간대: ${CURRENT_HOUR}시)"

    cd /home/intu/projects/pipe-inspector-electron

    # 기존 프로세스 확인
    if pgrep -f "convert_videos_to_web.py.*SAHARA" > /dev/null; then
        echo "[$(date)] SAHARA 변환 작업이 이미 실행 중입니다."
        exit 0
    fi

    # 변환 작업 시작 (기본 incremental 모드)
    nohup .venv/bin/python3 convert_videos_to_web.py --folder SAHARA > video_conversion_sahara.log 2>&1 &

    PID=$!
    echo "[$(date)] SAHARA 변환 작업 시작됨 (PID: $PID)"

    # 09시가 되면 자동 종료되도록 백그라운드 모니터링
    (
        while true; do
            sleep 300  # 5분마다 체크
            HOUR=$(date +%H)
            if [ "$HOUR" -ge 9 ]; then
                if kill -0 $PID 2>/dev/null; then
                    echo "[$(date)] 09시 도달, SAHARA 변환 작업 중지 (PID: $PID)"
                    kill $PID
                fi
                break
            fi
        done
    ) &

else
    echo "[$(date)] 현재 시간대(${CURRENT_HOUR}시)는 SAHARA 변환 실행 시간이 아닙니다. (01:00~09:00만 실행)"

    # 만약 실행 중인 프로세스가 있다면 종료
    SAHARA_PID=$(pgrep -f "convert_videos_to_web.py.*SAHARA")
    if [ -n "$SAHARA_PID" ]; then
        echo "[$(date)] 시간대 종료, SAHARA 변환 작업 중지 (PID: $SAHARA_PID)"
        kill $SAHARA_PID
    fi
fi
