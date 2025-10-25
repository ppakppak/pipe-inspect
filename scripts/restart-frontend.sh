#!/bin/bash

# Frontend (Renderer) 재시작 스크립트
# 실행 중인 Electron 앱의 Renderer를 재로드합니다.

echo "🔄 Reloading Frontend (Renderer)..."

# DevTools 콘솔에서 실행할 명령어를 전송
# Electron 창에 Ctrl+R (reload) 신호 전송
xdotool search --name "pipe-inspector-electron" windowactivate --sync key --clearmodifiers ctrl+r 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Frontend reloaded successfully"
else
    echo "⚠️  xdotool not found. Install it: sudo apt install xdotool"
    echo "   Or manually press Ctrl+R in the app window"
fi
