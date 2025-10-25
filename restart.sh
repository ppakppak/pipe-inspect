#!/bin/bash

# 스크립트 디렉토리 경로 설정
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Python 가상환경 경로
PYTHON="$SCRIPT_DIR/.venv/bin/python3"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Pipe Inspector - Restart Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 기존 프로세스 종료
echo -e "${YELLOW}[1/4] Stopping existing processes...${NC}"

# Backend Proxy 종료 (포트 5001)
BACKEND_PID=$(lsof -ti:5001 -sTCP:LISTEN)
if [ ! -z "$BACKEND_PID" ]; then
    echo -e "  ${RED}✖${NC} Killing Backend Proxy (PID: $BACKEND_PID)"
    kill -9 $BACKEND_PID 2>/dev/null
    sleep 2
else
    echo -e "  ${GREEN}✓${NC} No Backend Proxy running"
fi

# GPU Server 종료 (포트 5002)
GPU_PID=$(lsof -ti:5002 -sTCP:LISTEN)
if [ ! -z "$GPU_PID" ]; then
    echo -e "  ${RED}✖${NC} Killing GPU Server (PID: $GPU_PID)"
    kill -9 $GPU_PID 2>/dev/null
    sleep 2
else
    echo -e "  ${GREEN}✓${NC} No GPU Server running"
fi

# Python 프로세스 정리 (backend_proxy.py, api.py)
PYTHON_PIDS=$(pgrep -f "python.*backend_proxy.py|python.*api.py")
if [ ! -z "$PYTHON_PIDS" ]; then
    echo -e "  ${RED}✖${NC} Killing remaining Python processes"
    echo "$PYTHON_PIDS" | xargs kill -9 2>/dev/null
    sleep 1
fi

echo -e "${GREEN}✓ All processes stopped${NC}"
echo ""

# 2. 포트 확인
echo -e "${YELLOW}[2/4] Checking ports...${NC}"

# 포트 5001 확인
if lsof -ti:5001 -sTCP:LISTEN > /dev/null 2>&1; then
    echo -e "  ${RED}✖ Port 5001 still in use${NC}"
    exit 1
else
    echo -e "  ${GREEN}✓${NC} Port 5001 available"
fi

# 포트 5002 확인
if lsof -ti:5002 -sTCP:LISTEN > /dev/null 2>&1; then
    echo -e "  ${RED}✖ Port 5002 still in use${NC}"
    exit 1
else
    echo -e "  ${GREEN}✓${NC} Port 5002 available"
fi

echo ""

# 3. GPU 서버 시작
echo -e "${YELLOW}[3/4] Starting GPU Server...${NC}"

cd "$SCRIPT_DIR/gpu-server"

# 로그 파일 경로
GPU_LOG="$SCRIPT_DIR/gpu-server.log"

# GPU 서버 시작 (백그라운드)
nohup "$PYTHON" api.py > "$GPU_LOG" 2>&1 &
GPU_PID=$!

echo -e "  ${GREEN}✓${NC} GPU Server started (PID: $GPU_PID)"
echo -e "  ${BLUE}ℹ${NC} Log: $GPU_LOG"

# GPU 서버 시작 대기 (최대 5초)
for i in {1..10}; do
    if lsof -ti:5002 -sTCP:LISTEN > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} GPU Server is ready on port 5002"
        break
    fi
    sleep 0.5
done

echo ""

# 4. Backend Proxy 시작
echo -e "${YELLOW}[4/4] Starting Backend Proxy...${NC}"

cd "$SCRIPT_DIR"

# 로그 파일 경로
BACKEND_LOG="$SCRIPT_DIR/backend-proxy.log"

# Backend Proxy 시작 (백그라운드)
nohup "$PYTHON" backend_proxy.py > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

echo -e "  ${GREEN}✓${NC} Backend Proxy started (PID: $BACKEND_PID)"
echo -e "  ${BLUE}ℹ${NC} Log: $BACKEND_LOG"

# Backend Proxy 시작 대기 (최대 5초)
for i in {1..10}; do
    if lsof -ti:5001 -sTCP:LISTEN > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓${NC} Backend Proxy is ready on port 5001"
        break
    fi
    sleep 0.5
done

echo ""

# 5. 상태 확인
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  Restart Complete!${NC}"
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
