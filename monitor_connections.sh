#!/bin/bash
# 서버 접속 상태 모니터링 스크립트

clear
echo "=================================="
echo "🖥️  서버 접속 상태 모니터링"
echo "=================================="
echo ""

# 1. 시스템 부하
echo "📊 시스템 부하:"
uptime
echo ""

# 2. 현재 로그인 사용자
echo "👥 현재 로그인 사용자:"
who
echo ""

# 3. 백엔드 서버 상태 (Flask)
echo "🌐 백엔드 서버 상태:"
if pgrep -f "backend_proxy.py" > /dev/null; then
    echo "  ✅ backend_proxy.py 실행 중"
    netstat -an 2>/dev/null | grep -E ':(5003)' | grep LISTEN | head -1
else
    echo "  ❌ backend_proxy.py 중지됨"
fi
echo ""

# 4. GPU 서버 상태
echo "🎮 GPU 서버 상태:"
if pgrep -f "gpu-server.*api.py" > /dev/null; then
    echo "  ✅ GPU 서버 실행 중"
    netstat -an 2>/dev/null | grep -E ':(5004)' | grep LISTEN | head -1
else
    echo "  ❌ GPU 서버 중지됨"
fi
echo ""

# 5. 활성 웹 연결 수
echo "🔗 활성 웹 연결:"
BACKEND_CONN=$(netstat -an 2>/dev/null | grep -E ':(5003|5004)' | grep ESTABLISHED | wc -l)
echo "  현재 연결: ${BACKEND_CONN}개"
echo ""

# 6. 최근 접속 로그 (backend-proxy.log에서)
echo "📝 최근 접속 로그 (최근 5개):"
if [ -f "backend-proxy.log" ]; then
    grep "GET\|POST" backend-proxy.log | tail -5 | while read line; do
        echo "  $line"
    done
else
    echo "  로그 파일 없음"
fi
echo ""

# 7. 네트워크 통계
echo "📶 네트워크 통계:"
echo "  현재 연결된 IP 주소:"
netstat -an 2>/dev/null | grep ESTABLISHED | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -rn | head -5
echo ""

echo "=================================="
echo "모니터링 완료 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================="
