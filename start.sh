#!/bin/bash

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Pipe Inspector - Start Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 포트 확인
echo -e "${YELLOW}[1/3] Checking ports...${NC}"

# 포트 5001 확인
if lsof -ti:5001 > /dev/null 2>&1; then
    echo -e "  ${RED}✖ Port 5001 already in use${NC}"
    echo -e "  ${YELLOW}Run ./stop.sh first to stop existing services${NC}"
    exit 1
else
    echo -e "  ${GREEN}✓${NC} Port 5001 available"
fi

# 포트 5002 확인
if lsof -ti:5002 > /dev/null 2>&1; then
    echo -e "  ${RED}✖ Port 5002 already in use${NC}"
    echo -e "  ${YELLOW}Run ./stop.sh first to stop existing services${NC}"
    exit 1
else
    echo -e "  ${GREEN}✓${NC} Port 5002 available"
fi

echo ""

# 2. GPU 서버 시작
echo -e "${YELLOW}[2/3] Starting GPU Server...${NC}"

cd /home/ppak/pipe-inspector-electron/gpu-server

# 로그 파일 경로
GPU_LOG="/home/ppak/pipe-inspector-electron/gpu-server.log"

# GPU 서버 시작 (백그라운드)
nohup python3 api.py > "$GPU_LOG" 2>&1 &
GPU_PID=$!

echo -e "  ${GREEN}✓${NC} GPU Server started (PID: $GPU_PID)"
echo -e "  ${BLUE}ℹ${NC} Log: $GPU_LOG"

# GPU 서버 시작 대기 (최대 5초)
for i in {1..10}; do
    if lsof -ti:5002 > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} GPU Server is ready on port 5002"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "  ${RED}✖ GPU Server failed to start${NC}"
        echo -e "  ${YELLOW}Check log: tail -f $GPU_LOG${NC}"
        exit 1
    fi
    sleep 0.5
done

echo ""

# 3. Backend Proxy 시작
echo -e "${YELLOW}[3/3] Starting Backend Proxy...${NC}"

cd /home/ppak/pipe-inspector-electron

# 로그 파일 경로
BACKEND_LOG="/home/ppak/pipe-inspector-electron/backend-proxy.log"

# Backend Proxy 시작 (백그라운드)
nohup python3 backend_proxy.py > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

echo -e "  ${GREEN}✓${NC} Backend Proxy started (PID: $BACKEND_PID)"
echo -e "  ${BLUE}ℹ${NC} Log: $BACKEND_LOG"

# Backend Proxy 시작 대기 (최대 5초)
for i in {1..10}; do
    if lsof -ti:5001 > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Backend Proxy is ready on port 5001"
        break
    fi
    if [ $i -eq 10 ]; then
        echo -e "  ${RED}✖ Backend Proxy failed to start${NC}"
        echo -e "  ${YELLOW}Check log: tail -f $BACKEND_LOG${NC}"
        exit 1
    fi
    sleep 0.5
done

echo ""

# 4. 시작 완료
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Services Started Successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${BLUE}📡 Services:${NC}"
echo -e "  • Backend Proxy: ${GREEN}http://localhost:5001${NC}"
echo -e "  • GPU Server:    ${GREEN}http://localhost:5002${NC}"
echo ""

echo -e "${BLUE}👥 Default Account:${NC}"
echo -e "  • User ID:  ${YELLOW}admin${NC}"
echo -e "  • Password: ${YELLOW}admin123${NC}"
echo ""

echo -e "${BLUE}📋 Logs:${NC}"
echo -e "  • Backend: ${YELLOW}tail -f $BACKEND_LOG${NC}"
echo -e "  • GPU:     ${YELLOW}tail -f $GPU_LOG${NC}"
echo ""

echo -e "${BLUE}🌐 Open in browser:${NC}"
echo -e "  ${GREEN}http://localhost:5001${NC}"
echo ""

# 프로세스 상태 표시
echo -e "${BLUE}🔍 Process Status:${NC}"
ps aux | grep -E "python.*(backend_proxy|api\.py)" | grep -v grep | awk '{printf "  • PID %s: %s\n", $2, $11}'
echo ""

echo -e "${BLUE}🛑 To stop:${NC}"
echo -e "  ${YELLOW}./stop.sh${NC}"
echo ""
