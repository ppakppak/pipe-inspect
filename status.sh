#!/bin/bash

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Pipe Inspector - Status Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 포트 상태 확인
echo -e "${BLUE}🔌 Port Status:${NC}"

check_port() {
    local PORT=$1
    local NAME=$2

    if lsof -ti:$PORT > /dev/null 2>&1; then
        PID=$(lsof -ti:$PORT)
        PROCESS=$(ps -p $PID -o comm= 2>/dev/null || echo "Unknown")
        echo -e "  • Port $PORT ($NAME): ${GREEN}Running${NC} (PID: $PID, Process: $PROCESS)"
        return 0
    else
        echo -e "  • Port $PORT ($NAME): ${RED}Not running${NC}"
        return 1
    fi
}

BACKEND_STATUS=1
GPU_STATUS=1

check_port 5001 "Backend Proxy"
BACKEND_STATUS=$?

check_port 5002 "GPU Server"
GPU_STATUS=$?

echo ""

# 2. 프로세스 목록
echo -e "${BLUE}📋 Running Processes:${NC}"

PROCESSES=$(ps aux | grep -E "python.*(backend_proxy|api\.py)" | grep -v grep)
if [ ! -z "$PROCESSES" ]; then
    echo "$PROCESSES" | awk '{printf "  • PID %s: %s (CPU: %s%%, MEM: %s%%)\n", $2, $11, $3, $4}'
else
    echo -e "  ${RED}No processes running${NC}"
fi

echo ""

# 3. 서비스 URL
echo -e "${BLUE}🌐 Service URLs:${NC}"

if [ $BACKEND_STATUS -eq 0 ]; then
    echo -e "  • Backend Proxy: ${GREEN}http://localhost:5001${NC}"
else
    echo -e "  • Backend Proxy: ${RED}Not available${NC}"
fi

if [ $GPU_STATUS -eq 0 ]; then
    echo -e "  • GPU Server:    ${GREEN}http://localhost:5002${NC}"
else
    echo -e "  • GPU Server:    ${RED}Not available${NC}"
fi

echo ""

# 4. 로그 파일
echo -e "${BLUE}📄 Log Files:${NC}"

BACKEND_LOG="/home/ppak/pipe-inspector-electron/backend-proxy.log"
GPU_LOG="/home/ppak/pipe-inspector-electron/gpu-server.log"

if [ -f "$BACKEND_LOG" ]; then
    SIZE=$(du -h "$BACKEND_LOG" | cut -f1)
    MODIFIED=$(stat -c %y "$BACKEND_LOG" | cut -d. -f1)
    echo -e "  • Backend: ${YELLOW}$BACKEND_LOG${NC} ($SIZE, Modified: $MODIFIED)"
else
    echo -e "  • Backend: ${RED}No log file${NC}"
fi

if [ -f "$GPU_LOG" ]; then
    SIZE=$(du -h "$GPU_LOG" | cut -f1)
    MODIFIED=$(stat -c %y "$GPU_LOG" | cut -d. -f1)
    echo -e "  • GPU:     ${YELLOW}$GPU_LOG${NC} ($SIZE, Modified: $MODIFIED)"
else
    echo -e "  • GPU:     ${RED}No log file${NC}"
fi

echo ""

# 5. 시스템 리소스
echo -e "${BLUE}💻 System Resources:${NC}"

# CPU 사용량
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d% -f1)
echo -e "  • CPU Usage: ${YELLOW}${CPU_USAGE}%${NC}"

# 메모리 사용량
MEM_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo -e "  • Memory Usage: ${YELLOW}${MEM_USAGE}%${NC}"

# GPU 메모리 (nvidia-smi가 있는 경우)
if command -v nvidia-smi &> /dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "%.1f", $1/$2*100}')
    GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    echo -e "  • GPU Memory: ${YELLOW}${GPU_MEM_USED}MB / ${GPU_MEM_TOTAL}MB (${GPU_MEM}%)${NC}"
fi

echo ""

# 6. 전체 상태 요약
echo -e "${BLUE}========================================${NC}"

if [ $BACKEND_STATUS -eq 0 ] && [ $GPU_STATUS -eq 0 ]; then
    echo -e "${GREEN}  ✓ All Services Running${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${BLUE}🚀 Quick Actions:${NC}"
    echo -e "  • Open browser:  ${GREEN}http://localhost:5001${NC}"
    echo -e "  • View logs:     ${YELLOW}tail -f $BACKEND_LOG${NC}"
    echo -e "  • Stop services: ${YELLOW}./stop.sh${NC}"
    echo -e "  • Restart:       ${YELLOW}./restart.sh${NC}"
elif [ $BACKEND_STATUS -ne 0 ] && [ $GPU_STATUS -ne 0 ]; then
    echo -e "${RED}  ✖ All Services Stopped${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${BLUE}🚀 Quick Actions:${NC}"
    echo -e "  • Start services: ${YELLOW}./start.sh${NC}"
else
    echo -e "${YELLOW}  ⚠ Partial Service Running${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${BLUE}🚀 Quick Actions:${NC}"
    echo -e "  • Restart:       ${YELLOW}./restart.sh${NC}"
    echo -e "  • Stop all:      ${YELLOW}./stop.sh${NC}"
fi

echo ""
