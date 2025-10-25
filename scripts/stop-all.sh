#!/bin/bash

# 전체 앱 종료 스크립트

echo "🛑 Stopping all services..."

# Backend 종료
echo "   Stopping Backend..."
pkill -f "python.*backend.py" 2>/dev/null && echo "   ✅ Backend stopped" || echo "   ℹ️  Backend not running"

# Frontend 종료
echo "   Stopping Frontend..."
pkill -f "electron.*pipe-inspector-electron" 2>/dev/null && echo "   ✅ Frontend stopped" || echo "   ℹ️  Frontend not running"

echo ""
echo "✅ All services stopped"
