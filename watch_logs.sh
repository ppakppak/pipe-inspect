#!/bin/bash
# 여러 로그 파일을 동시에 실시간으로 모니터링

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# 사용법 표시
show_usage() {
    echo "사용법: $0 [옵션]"
    echo ""
    echo "옵션:"
    echo "  backend    - 백엔드 로그만 보기"
    echo "  gpu        - GPU 서버 로그만 보기"
    echo "  convert    - 비디오 변환 로그만 보기"
    echo "  all        - 모든 로그 함께 보기 (기본값)"
    echo "  errors     - 에러만 필터링해서 보기"
    echo ""
    echo "예시:"
    echo "  $0 backend"
    echo "  $0 all"
    echo "  $0 errors"
}

# 로그 파일 경로
BACKEND_LOG="backend-proxy.log"
GPU_LOG="gpu-server/gpu-server.log"
CONVERT_LOG="video_conversion_kwanrae.log"
CONVERT_MANUAL_LOG="video_conversion_manual.log"

# 파라미터 처리
MODE=${1:-all}

case $MODE in
    backend)
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}📊 백엔드 로그 실시간 모니터링${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        if [ -f "$BACKEND_LOG" ]; then
            tail -f "$BACKEND_LOG" | while read line; do
                if echo "$line" | grep -qi "error"; then
                    echo -e "${RED}${line}${NC}"
                elif echo "$line" | grep -qi "warning"; then
                    echo -e "${YELLOW}${line}${NC}"
                elif echo "$line" | grep -qi "info"; then
                    echo -e "${GREEN}${line}${NC}"
                else
                    echo "$line"
                fi
            done
        else
            echo -e "${RED}로그 파일을 찾을 수 없습니다: $BACKEND_LOG${NC}"
        fi
        ;;

    gpu)
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}🎮 GPU 서버 로그 실시간 모니터링${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        if [ -f "$GPU_LOG" ]; then
            tail -f "$GPU_LOG" | while read line; do
                if echo "$line" | grep -qi "error"; then
                    echo -e "${RED}${line}${NC}"
                elif echo "$line" | grep -qi "warning"; then
                    echo -e "${YELLOW}${line}${NC}"
                elif echo "$line" | grep -qi "info"; then
                    echo -e "${CYAN}${line}${NC}"
                else
                    echo "$line"
                fi
            done
        else
            echo -e "${RED}로그 파일을 찾을 수 없습니다: $GPU_LOG${NC}"
        fi
        ;;

    convert)
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}📹 비디오 변환 로그 실시간 모니터링${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        # 최신 변환 로그 찾기
        if [ -f "$CONVERT_LOG" ]; then
            LOG_FILE="$CONVERT_LOG"
        elif [ -f "$CONVERT_MANUAL_LOG" ]; then
            LOG_FILE="$CONVERT_MANUAL_LOG"
        else
            echo -e "${RED}변환 로그 파일을 찾을 수 없습니다${NC}"
            exit 1
        fi

        echo -e "로그 파일: ${CYAN}${LOG_FILE}${NC}"
        echo ""

        tail -f "$LOG_FILE" | while read line; do
            if echo "$line" | grep -qi "error\|실패\|failed"; then
                echo -e "${RED}${line}${NC}"
            elif echo "$line" | grep -qi "warning\|타임아웃"; then
                echo -e "${YELLOW}${line}${NC}"
            elif echo "$line" | grep -qi "변환 완료\|completed"; then
                echo -e "${GREEN}${line}${NC}"
            elif echo "$line" | grep -qi "진행:"; then
                echo -e "${CYAN}${line}${NC}"
            else
                echo "$line"
            fi
        done
        ;;

    errors)
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}❌ 에러 로그 실시간 모니터링${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""

        # 모든 로그 파일을 동시에 모니터링하고 에러만 표시
        (
            [ -f "$BACKEND_LOG" ] && tail -f "$BACKEND_LOG" | grep -i "error" | sed 's/^/[BACKEND] /' &
            [ -f "$GPU_LOG" ] && tail -f "$GPU_LOG" | grep -i "error" | sed 's/^/[GPU] /' &
            [ -f "$CONVERT_LOG" ] && tail -f "$CONVERT_LOG" | grep -iE "error|실패|failed" | sed 's/^/[CONVERT] /' &
            wait
        ) | while read line; do
            echo -e "${RED}${line}${NC}"
        done
        ;;

    all)
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}📊 전체 로그 실시간 모니터링${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        echo -e "${YELLOW}Ctrl+C를 눌러 종료${NC}"
        echo ""

        # 모든 로그 파일을 동시에 모니터링
        (
            if [ -f "$BACKEND_LOG" ]; then
                tail -f "$BACKEND_LOG" | sed 's/^/[BACKEND] /' | while read line; do
                    if echo "$line" | grep -qi "error"; then
                        echo -e "${RED}${line}${NC}"
                    else
                        echo -e "${GREEN}${line}${NC}"
                    fi
                done &
            fi

            if [ -f "$GPU_LOG" ]; then
                tail -f "$GPU_LOG" | sed 's/^/[GPU] /' | while read line; do
                    if echo "$line" | grep -qi "error"; then
                        echo -e "${RED}${line}${NC}"
                    else
                        echo -e "${CYAN}${line}${NC}"
                    fi
                done &
            fi

            if [ -f "$CONVERT_LOG" ] || [ -f "$CONVERT_MANUAL_LOG" ]; then
                LOG_FILE="$CONVERT_LOG"
                [ ! -f "$LOG_FILE" ] && LOG_FILE="$CONVERT_MANUAL_LOG"

                tail -f "$LOG_FILE" 2>/dev/null | sed 's/^/[CONVERT] /' | while read line; do
                    if echo "$line" | grep -qi "error\|실패"; then
                        echo -e "${RED}${line}${NC}"
                    elif echo "$line" | grep -qi "변환 완료"; then
                        echo -e "${MAGENTA}${line}${NC}"
                    else
                        echo -e "${YELLOW}${line}${NC}"
                    fi
                done &
            fi

            wait
        )
        ;;

    -h|--help|help)
        show_usage
        exit 0
        ;;

    *)
        echo -e "${RED}알 수 없는 옵션: $MODE${NC}"
        echo ""
        show_usage
        exit 1
        ;;
esac
