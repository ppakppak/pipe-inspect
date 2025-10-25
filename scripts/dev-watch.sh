#!/bin/bash

# 개발 모드: 파일 변경 감지 후 자동 재시작
# 필요 패키지: inotify-tools (sudo apt install inotify-tools)

cd /home/ppak/pipe-inspector-electron

echo "👀 Watching for file changes..."
echo "   backend.py → Backend restart"
echo "   main.js → Electron Main Process restart"
echo "   *.html, *.css → Renderer reload"
echo ""

# 초기 실행
./scripts/start-all.sh

# 파일 감시
inotifywait -m -r -e modify --format '%w%f' \
  --exclude 'node_modules|\.git|scripts|backend\.log' \
  /home/ppak/pipe-inspector-electron/ | while read FILE
do
  echo "🔍 Changed: $FILE"

  if [[ "$FILE" == *"backend.py"* ]]; then
    echo "🔄 Backend changed - Restarting Python server"
    ./scripts/restart-backend.sh
  elif [[ "$FILE" == *"main.js"* ]]; then
    echo "🔄 Main process changed - Restarting Electron"
    ./scripts/restart-main.sh
  elif [[ "$FILE" == *".html"* ]] || [[ "$FILE" == *".css"* ]]; then
    echo "🔄 Frontend changed - Reloading renderer"
    xdotool search --name "pipe-inspector-electron" windowactivate --sync key --clearmodifiers ctrl+r 2>/dev/null || echo "   (xdotool not available - press Ctrl+R manually)"
  fi
done
