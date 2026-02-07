#!/bin/bash
# 네트워크 사용량을 실시간으로 모니터링하고 어떤 프로세스가 사용 중인지 확인

# 색상 정의
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🌐 네트워크 활동 모니터링${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 1. 전체 네트워크 인터페이스별 사용량
echo -e "${YELLOW}📊 네트워크 인터페이스 사용량:${NC}"
ifconfig 2>/dev/null | grep -E "^[a-z]|RX|TX" | grep -v "127.0.0.1" | head -20
echo ""

# 2. 현재 활성 연결 (프로세스별)
echo -e "${YELLOW}🔗 활성 네트워크 연결 (프로세스별):${NC}"
echo "PID        프로세스              로컬주소              원격주소              상태"
echo "--------------------------------------------------------------------------------"
netstat -tunap 2>/dev/null | grep ESTABLISHED | awk '{
    if ($7 != "-") {
        split($7, a, "/");
        pid = a[1];
        # 프로세스 이름 가져오기
        cmd = "ps -p " pid " -o comm= 2>/dev/null";
        cmd | getline proc;
        close(cmd);
        printf "%-10s %-20s %-20s %-20s %-12s\n", pid, substr(proc, 1, 20), substr($4, 1, 20), substr($5, 1, 20), $6;
    }
}' | head -20
echo ""

# 3. 프로세스별 네트워크 사용량 (업로드/다운로드)
echo -e "${YELLOW}📈 프로세스별 네트워크 사용량 (대역폭):${NC}"
if command -v nethogs &> /dev/null; then
    echo "nethogs가 설치되어 있습니다. sudo nethogs를 실행하세요."
else
    echo "더 자세한 분석을 위해 nethogs 설치를 권장합니다:"
    echo "  sudo apt install nethogs"
    echo "  sudo nethogs"
fi
echo ""

# 4. 가장 많은 연결을 가진 프로세스
echo -e "${YELLOW}🔝 가장 많은 연결을 가진 프로세스 TOP 5:${NC}"
netstat -tunap 2>/dev/null | grep ESTABLISHED | awk '{print $7}' | grep -v "-" | cut -d/ -f1 | sort | uniq -c | sort -rn | head -5 | while read count pid; do
    proc=$(ps -p $pid -o comm= 2>/dev/null)
    echo "  $count개 연결: $proc (PID: $pid)"
done
echo ""

# 5. 외부 IP별 연결 통계
echo -e "${YELLOW}🌍 외부 IP별 연결 통계:${NC}"
netstat -tunap 2>/dev/null | grep ESTABLISHED | awk '{print $5}' | cut -d: -f1 | grep -v "127.0.0.1" | sort | uniq -c | sort -rn | head -10 | while read count ip; do
    echo "  $count개 연결: $ip"
done
echo ""

# 6. 포트별 연결 통계
echo -e "${YELLOW}🔌 로컬 포트별 리스닝 서비스:${NC}"
netstat -tuln 2>/dev/null | grep LISTEN | awk '{print $4}' | rev | cut -d: -f1 | rev | sort -n | uniq | while read port; do
    proc=$(lsof -i :$port 2>/dev/null | tail -1 | awk '{print $1}')
    [ -n "$proc" ] && echo "  포트 $port: $proc"
done | head -10
echo ""

# 7. 특정 프로세스의 네트워크 활동 (Claude, Chrome, 등)
echo -e "${YELLOW}🎯 주요 프로세스별 연결:${NC}"
for proc in "claude" "chrome" "backend_proxy" "api.py"; do
    pid=$(pgrep -f "$proc" | head -1)
    if [ -n "$pid" ]; then
        count=$(netstat -tunap 2>/dev/null | grep "$pid" | grep ESTABLISHED | wc -l)
        if [ $count -gt 0 ]; then
            echo -e "  ${CYAN}$proc${NC} (PID: $pid): $count개 연결"
            netstat -tunap 2>/dev/null | grep "$pid" | grep ESTABLISHED | awk '{print "    → " $5}' | head -3
        fi
    fi
done
echo ""

echo -e "${BLUE}========================================${NC}"
echo "실시간 모니터링: watch -n 1 ./monitor_network.sh"
echo "또는 iftop 설치: sudo apt install iftop && sudo iftop"
echo -e "${BLUE}========================================${NC}"
