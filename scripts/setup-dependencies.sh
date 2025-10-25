#!/bin/bash

# 의존성 설치 스크립트

cd /home/ppak/pipe-inspector-electron

echo "📦 Installing Python dependencies..."

# Backend (클라이언트) 의존성
echo "1️⃣ Installing Backend (MCP Client) dependencies..."
pip3 install -r requirements.txt

# MCP Server (GPU 서버) 의존성
echo "2️⃣ Installing MCP Server dependencies..."
pip3 install -r mcp-server/requirements.txt

echo "✅ All dependencies installed!"
