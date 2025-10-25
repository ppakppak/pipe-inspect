#!/bin/bash

# Python Backend 시작 스크립트 (MCP Client Mode)

cd /home/ppak/pipe-inspector-electron

echo "🐍 Starting Python Backend (MCP Client Mode)..."
echo "📡 API Server will be available at: http://localhost:5003"
echo "🔌 Connecting to MCP Server..."

# 기존 프로세스 종료
pkill -f "python.*backend.py" 2>/dev/null

# 로그 파일 초기화
> backend.log

# 백엔드 실행 (conda 환경 사용, 백그라운드)
conda run -n mcp-server python3 backend.py >> backend.log 2>&1 &
BACKEND_PID=$!

echo "✅ Backend started (PID: $BACKEND_PID)"
echo "📄 Logs: tail -f backend.log"

# 서버가 준비될 때까지 대기
sleep 3

# 헬스 체크
if curl -s http://localhost:5003/api/health > /dev/null 2>&1; then
    echo "✅ Backend health check passed"
    echo "✅ MCP Server connection verified"
else
    echo "⚠️  Backend may not be ready yet. Check logs: tail -f backend.log"
fi
