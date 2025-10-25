#!/bin/bash

# 전체 앱 재시작 스크립트 (Backend + Frontend)

cd /home/ppak/pipe-inspector-electron

echo "🔄 Restarting all services..."
echo ""

# 모든 서비스 종료
./scripts/stop-all.sh

echo ""
sleep 1

# 모든 서비스 시작
./scripts/start-all.sh
