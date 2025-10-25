#!/bin/bash

# Electron Frontend 시작 스크립트

cd /home/ppak/pipe-inspector-electron

echo "⚛️  Starting Electron Frontend..."

# 기존 프로세스 종료
pkill -f "electron.*pipe-inspector-electron" 2>/dev/null

sleep 1

# 환경 변수 정리 (Electron을 Node 모드로 실행하지 않도록)
unset ELECTRON_RUN_AS_NODE

# 프론트엔드 실행
npm start &

echo "✅ Frontend started"
