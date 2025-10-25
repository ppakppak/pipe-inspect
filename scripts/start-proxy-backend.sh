#!/bin/bash

# Proxy Backend 시작 스크립트 (클라이언트 PC용)

cd /home/ppak/pipe-inspector-electron

# GPU 서버 URL (환경 변수 또는 기본값)
GPU_SERVER_URL=${GPU_SERVER_URL:-http://localhost:5004}

echo "🔌 Starting Proxy Backend..."
echo "📡 Client API: http://localhost:5003"
echo "🎮 GPU Server: $GPU_SERVER_URL"
echo ""

# 기존 프로세스 종료
pkill -f "python.*backend_proxy.py" 2>/dev/null

# 프록시 백엔드 실행
python3 backend_proxy.py
