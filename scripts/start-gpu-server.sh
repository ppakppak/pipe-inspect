#!/bin/bash

# GPU Server 시작 스크립트

cd /home/ppak/pipe-inspector-electron

echo "🎮 Starting GPU Server..."
echo "📡 API Server: http://0.0.0.0:5004"
echo ""

# GPU 서버 실행
python3 gpu-server/api.py
