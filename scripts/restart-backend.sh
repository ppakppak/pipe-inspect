#!/bin/bash

# Python Backend 재시작 스크립트
# backend.py가 변경되었을 때 사용

cd /home/ppak/pipe-inspector-electron

echo "🔄 Restarting Python Backend..."

# 기존 프로세스 종료
pkill -f "python.*backend.py" 2>/dev/null

sleep 1

# 백엔드 재시작
./scripts/start-backend.sh
