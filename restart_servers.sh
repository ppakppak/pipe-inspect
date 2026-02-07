#!/bin/bash

# Pipe Inspector 서버 재시작 스크립트
# 사용법: ./restart_servers.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv/bin/activate"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Pipe Inspector 서버 재시작 스크립트${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 기존 서버 종료
echo -e "${YELLOW}[1/4] 기존 서버 종료 중...${NC}"

# Backend Proxy 종료
if pgrep -f "python.*backend_proxy.py" > /dev/null; then
    pkill -f "python.*backend_proxy.py"
    echo -e "  ${GREEN}✓${NC} Backend Proxy 종료됨"
else
    echo -e "  ${YELLOW}-${NC} Backend Proxy 실행 중 아님"
fi

# GPU Server 종료
if pgrep -f "python.*gpu-server/api.py" > /dev/null; then
    pkill -f "python.*gpu-server/api.py"
    echo -e "  ${GREEN}✓${NC} GPU Server 종료됨"
else
    echo -e "  ${YELLOW}-${NC} GPU Server 실행 중 아님"
fi

# 프로세스 종료 대기
sleep 2

# 2. 포트 확인
echo ""
echo -e "${YELLOW}[2/4] 포트 상태 확인 중...${NC}"

check_port() {
    local port=$1
    local name=$2
    if lsof -i :$port > /dev/null 2>&1; then
        echo -e "  ${RED}✗${NC} 포트 $port ($name) 사용 중 - 강제 종료 시도"
        fuser -k $port/tcp 2>/dev/null
        sleep 1
    else
        echo -e "  ${GREEN}✓${NC} 포트 $port ($name) 사용 가능"
    fi
}

check_port 5003 "Backend Proxy"
check_port 5004 "GPU Server"

# 3. 서버 시작
echo ""
echo -e "${YELLOW}[3/4] 서버 시작 중...${NC}"

# 가상환경 활성화 확인
if [ ! -f "$VENV_PATH" ]; then
    echo -e "  ${RED}✗${NC} 가상환경을 찾을 수 없습니다: $VENV_PATH"
    exit 1
fi

# Backend Proxy 시작
echo -e "  ${BLUE}→${NC} Backend Proxy 시작 중..."
cd "$SCRIPT_DIR"
source "$VENV_PATH"
nohup python backend_proxy.py > /tmp/backend_proxy.log 2>&1 &
BACKEND_PID=$!
sleep 2

if ps -p $BACKEND_PID > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Backend Proxy 시작됨 (PID: $BACKEND_PID)"
else
    echo -e "  ${RED}✗${NC} Backend Proxy 시작 실패"
    echo -e "  ${RED}  로그 확인: cat /tmp/backend_proxy.log${NC}"
fi

# GPU Server 시작
echo -e "  ${BLUE}→${NC} GPU Server 시작 중..."
cd "$SCRIPT_DIR/gpu-server"
nohup python api.py > /tmp/gpu_server.log 2>&1 &
GPU_PID=$!
sleep 3

if ps -p $GPU_PID > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} GPU Server 시작됨 (PID: $GPU_PID)"
else
    echo -e "  ${RED}✗${NC} GPU Server 시작 실패"
    echo -e "  ${RED}  로그 확인: cat /tmp/gpu_server.log${NC}"
fi

# 4. 상태 확인
echo ""
echo -e "${YELLOW}[4/4] 서버 상태 확인...${NC}"
sleep 1

echo ""
echo -e "${BLUE}서버 상태:${NC}"
echo -e "  Backend Proxy (5003): $(lsof -i :5003 > /dev/null 2>&1 && echo -e "${GREEN}실행 중${NC}" || echo -e "${RED}중지됨${NC}")"
echo -e "  GPU Server    (5004): $(lsof -i :5004 > /dev/null 2>&1 && echo -e "${GREEN}실행 중${NC}" || echo -e "${RED}중지됨${NC}")"

echo ""
echo -e "${BLUE}로그 파일:${NC}"
echo -e "  Backend Proxy: /tmp/backend_proxy.log"
echo -e "  GPU Server:    /tmp/gpu_server.log"

echo ""
echo -e "${BLUE}접속 주소:${NC}"
echo -e "  http://localhost:5003"
echo -e "  http://$(hostname -I | awk '{print $1}'):5003"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   서버 재시작 완료!${NC}"
echo -e "${GREEN}========================================${NC}"
