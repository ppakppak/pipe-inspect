#!/bin/bash

# Electron Main Process 재시작 스크립트
# main.js가 변경되었을 때 사용

cd /home/ppak/pipe-inspector-electron

echo "🔄 Restarting Electron Main Process..."

# 기존 프로세스 종료
pkill -f "electron.*pipe-inspector-electron" 2>/dev/null

sleep 1

# 프론트엔드 재시작
./scripts/start-frontend.sh
