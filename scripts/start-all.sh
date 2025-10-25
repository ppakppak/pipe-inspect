#!/bin/bash

# 전체 앱 시작 스크립트 (Backend + Frontend)

cd /home/ppak/pipe-inspector-electron

echo "🚀 Starting Pipe Inspector..."
echo ""

# Backend 시작
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./scripts/start-backend.sh
echo ""

# Frontend 시작
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./scripts/start-frontend.sh
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All services started!"
echo ""
echo "📊 Status:"
echo "   Backend:  http://localhost:5003"
echo "   Frontend: Electron app window"
echo ""
echo "📄 Logs:"
echo "   Backend:  tail -f backend.log"
echo ""
echo "🛑 Stop all:"
echo "   ./scripts/stop-all.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
