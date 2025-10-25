#!/bin/bash

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Pipe Inspector - Stop Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 프로세스 종료 함수
stop_process() {
    local PORT=$1
    local NAME=$2

    PID=$(lsof -ti:$PORT)
    if [ ! -z "$PID" ]; then
        echo -e "  ${RED}✖${NC} Stopping $NAME (PID: $PID, Port: $PORT)"
        kill -9 $PID 2>/dev/null
        sleep 0.5

        # 종료 확인
        if lsof -ti:$PORT > /dev/null 2>&1; then
            echo -e "    ${RED}Failed to stop $NAME${NC}"
            return 1
        else
            echo -e "    ${GREEN}✓ $NAME stopped${NC}"
            return 0
        fi
    else
        echo -e "  ${BLUE}ℹ${NC} $NAME not running (Port: $PORT)"
        return 0
    fi
}

# 1. Backend Proxy 종료 (포트 5001)
echo -e "${YELLOW}[1/3] Stopping Backend Proxy...${NC}"
stop_process 5001 "Backend Proxy"
echo ""

# 2. GPU Server 종료 (포트 5002)
echo -e "${YELLOW}[2/3] Stopping GPU Server...${NC}"
stop_process 5002 "GPU Server"
echo ""

# 3. 남은 Python 프로세스 정리
echo -e "${YELLOW}[3/3] Cleaning up remaining processes...${NC}"

PYTHON_PIDS=$(pgrep -f "python.*backend_proxy.py|python.*api.py")
if [ ! -z "$PYTHON_PIDS" ]; then
    echo -e "  ${RED}✖${NC} Killing remaining Python processes:"
    echo "$PYTHON_PIDS" | while read pid; do
        PROCESS=$(ps -p $pid -o comm=)
        echo -e "    • PID $pid: $PROCESS"
    done
    echo "$PYTHON_PIDS" | xargs kill -9 2>/dev/null
    sleep 0.5
    echo -e "    ${GREEN}✓ Cleanup complete${NC}"
else
    echo -e "  ${GREEN}✓${NC} No remaining processes"
fi

echo ""

# 최종 상태 확인
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  All Services Stopped${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 포트 확인
echo -e "${BLUE}🔍 Port Status:${NC}"
if lsof -ti:5001 > /dev/null 2>&1; then
    echo -e "  • Port 5001: ${RED}In use${NC}"
else
    echo -e "  • Port 5001: ${GREEN}Available${NC}"
fi

if lsof -ti:5002 > /dev/null 2>&1; then
    echo -e "  • Port 5002: ${RED}In use${NC}"
else
    echo -e "  • Port 5002: ${GREEN}Available${NC}"
fi

echo ""

# 남은 프로세스 확인
REMAINING=$(ps aux | grep -E "python.*(backend_proxy|api\.py)" | grep -v grep)
if [ ! -z "$REMAINING" ]; then
    echo -e "${YELLOW}⚠ Warning: Some processes still running:${NC}"
    echo "$REMAINING" | awk '{printf "  • PID %s: %s\n", $2, $11}'
    echo ""
fi
