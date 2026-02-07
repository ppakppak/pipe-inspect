#!/bin/bash
# 실시간 서버 접속 모니터링 (1초마다 갱신)

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

while true; do
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}🔴 실시간 서버 모니터링${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 시스템 부하
    echo -e "${YELLOW}📊 시스템 부하:${NC}"
    uptime | awk -F'load average:' '{print "  " $2}'
    echo ""

    # 서버 상태
    echo -e "${YELLOW}🖥️  서버 상태:${NC}"
    if pgrep -f "backend_proxy.py" > /dev/null; then
        echo -e "  ${GREEN}✅ Backend (5003): RUNNING${NC}"
    else
        echo -e "  ${RED}❌ Backend (5003): STOPPED${NC}"
    fi

    if pgrep -f "gpu-server.*api.py" > /dev/null; then
        echo -e "  ${GREEN}✅ GPU Server (5004): RUNNING${NC}"
    else
        echo -e "  ${RED}❌ GPU Server (5004): STOPPED${NC}"
    fi
    echo ""

    # 활성 연결
    echo -e "${YELLOW}🔗 활성 연결:${NC}"
    CONN_5003=$(netstat -an 2>/dev/null | grep ':5003' | grep ESTABLISHED | wc -l)
    CONN_5004=$(netstat -an 2>/dev/null | grep ':5004' | grep ESTABLISHED | wc -l)
    TOTAL_CONN=$((CONN_5003 + CONN_5004))

    echo "  Backend (5003): ${CONN_5003}개"
    echo "  GPU Server (5004): ${CONN_5004}개"
    echo -e "  ${GREEN}총 연결: ${TOTAL_CONN}개${NC}"
    echo ""

    # 현재 접속 IP
    echo -e "${YELLOW}🌐 접속 중인 IP:${NC}"
    netstat -an 2>/dev/null | grep -E ':(5003|5004)' | grep ESTABLISHED | awk '{print $5}' | cut -d: -f1 | sort | uniq -c | sort -rn | while read count ip; do
        echo "  ${ip}: ${count}개 연결"
    done
    [ $TOTAL_CONN -eq 0 ] && echo "  (현재 접속자 없음)"
    echo ""

    # 최근 활동
    echo -e "${YELLOW}📝 최근 요청 (마지막 3개):${NC}"
    if [ -f "backend-proxy.log" ]; then
        tail -100 backend-proxy.log | grep -E 'GET|POST' | tail -3 | while read line; do
            echo "  $(echo $line | cut -d'-' -f1-3 | cut -c1-70)"
        done
    else
        echo "  로그 없음"
    fi
    echo ""

    echo -e "${BLUE}========================================${NC}"
    echo "Ctrl+C를 눌러 종료"

    sleep 1
done
