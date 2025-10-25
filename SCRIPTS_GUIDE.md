# 🚀 Pipe Inspector - 서버 관리 스크립트 가이드

자동화된 서버 시작/중지/재시작 스크립트를 사용하여 백엔드와 GPU 서버를 쉽게 관리할 수 있습니다.

---

## 📋 스크립트 목록

### 1. `start.sh` - 서버 시작
백엔드 프록시와 GPU 서버를 시작합니다.

### 2. `stop.sh` - 서버 중지
실행 중인 모든 서버를 안전하게 종료합니다.

### 3. `restart.sh` - 서버 재시작
모든 서버를 강제로 중지하고 다시 시작합니다.

### 4. `status.sh` - 상태 확인
현재 서버 상태, 프로세스, 리소스 사용량을 확인합니다.

---

## 🎯 사용 방법

### 서버 시작

```bash
cd /home/ppak/pipe-inspector-electron
./start.sh
```

**출력 예시**:
```
========================================
  Pipe Inspector - Start Script
========================================

[1/3] Checking ports...
  ✓ Port 5001 available
  ✓ Port 5002 available

[2/3] Starting GPU Server...
  ✓ GPU Server started (PID: 12345)
  ℹ Log: /home/ppak/pipe-inspector-electron/gpu-server.log
  ✓ GPU Server is ready on port 5002

[3/3] Starting Backend Proxy...
  ✓ Backend Proxy started (PID: 12346)
  ℹ Log: /home/ppak/pipe-inspector-electron/backend-proxy.log
  ✓ Backend Proxy is ready on port 5001

========================================
  Services Started Successfully!
========================================

📡 Services:
  • Backend Proxy: http://localhost:5001
  • GPU Server:    http://localhost:5002

👥 Default Account:
  • User ID:  admin
  • Password: admin123
```

### 서버 중지

```bash
./stop.sh
```

**출력 예시**:
```
========================================
  Pipe Inspector - Stop Script
========================================

[1/3] Stopping Backend Proxy...
  ✖ Stopping Backend Proxy (PID: 12346, Port: 5001)
    ✓ Backend Proxy stopped

[2/3] Stopping GPU Server...
  ✖ Stopping GPU Server (PID: 12345, Port: 5002)
    ✓ GPU Server stopped

[3/3] Cleaning up remaining processes...
  ✓ No remaining processes

========================================
  All Services Stopped
========================================
```

### 서버 재시작 (강제)

```bash
./restart.sh
```

**기능**:
- 기존 서버를 강제로 종료
- 포트 사용 중이어도 강제로 해제
- GPU 서버와 백엔드를 순차적으로 재시작

**사용 시기**:
- 코드 변경 후 재시작이 필요할 때
- 서버가 응답하지 않을 때
- 포트 충돌이 발생했을 때

### 상태 확인

```bash
./status.sh
```

**출력 예시**:
```
========================================
  Pipe Inspector - Status Check
========================================

🔌 Port Status:
  • Port 5001 (Backend Proxy): Running (PID: 12346)
  • Port 5002 (GPU Server): Running (PID: 12345)

📋 Running Processes:
  • PID 12345: python3 (CPU: 70.4%, MEM: 2.5%)
  • PID 12346: python3 (CPU: 1.2%, MEM: 0.1%)

🌐 Service URLs:
  • Backend Proxy: http://localhost:5001
  • GPU Server:    http://localhost:5002

📄 Log Files:
  • Backend: /home/ppak/pipe-inspector-electron/backend-proxy.log (4.0K)
  • GPU:     /home/ppak/pipe-inspector-electron/gpu-server.log (4.0K)

💻 System Resources:
  • CPU Usage: 15.2%
  • Memory Usage: 45.3%
  • GPU Memory: 4236MB / 24564MB (17.2%)

========================================
  ✓ All Services Running
========================================

🚀 Quick Actions:
  • Open browser:  http://localhost:5001
  • View logs:     tail -f /home/ppak/pipe-inspector-electron/backend-proxy.log
  • Stop services: ./stop.sh
  • Restart:       ./restart.sh
```

---

## 📝 로그 파일

### Backend Proxy 로그
```bash
tail -f /home/ppak/pipe-inspector-electron/backend-proxy.log
```

**로그 내용**:
- API 요청/응답
- 인증 이벤트 (로그인/로그아웃)
- 프로젝트 생성/조회
- 에러 메시지

### GPU 서버 로그
```bash
tail -f /home/ppak/pipe-inspector-electron/gpu-server.log
```

**로그 내용**:
- AI 모델 로드
- 추론 요청/결과
- GPU 메모리 사용량
- 에러 메시지

---

## 🔧 트러블슈팅

### 문제 1: 포트가 이미 사용 중
```
✖ Port 5001 already in use
```

**해결 방법**:
```bash
# 먼저 중지
./stop.sh

# 또는 강제 재시작
./restart.sh
```

### 문제 2: 서버가 시작되지 않음
```
✖ Backend Proxy failed to start
```

**해결 방법**:
```bash
# 로그 확인
tail -f /home/ppak/pipe-inspector-electron/backend-proxy.log

# 수동으로 시작해서 에러 확인
python3 backend_proxy.py
```

### 문제 3: 프로세스가 종료되지 않음
```bash
# 수동으로 프로세스 종료
lsof -ti:5001 | xargs kill -9
lsof -ti:5002 | xargs kill -9
```

### 문제 4: GPU 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# 모든 프로세스 종료
./stop.sh

# 재시작
./start.sh
```

---

## ⚙️ 고급 사용법

### 백그라운드 실행 확인
```bash
# 프로세스 확인
ps aux | grep python | grep -E "(backend_proxy|api\.py)"

# 포트 확인
lsof -i:5001
lsof -i:5002
```

### 로그 실시간 모니터링 (여러 창)
```bash
# 터미널 1: Backend 로그
tail -f backend-proxy.log

# 터미널 2: GPU 서버 로그
tail -f gpu-server.log

# 터미널 3: 시스템 리소스
watch -n 1 './status.sh'
```

### 서비스 자동 시작 (systemd)

**1. systemd 서비스 파일 생성**:

`/etc/systemd/system/pipe-inspector.service`:
```ini
[Unit]
Description=Pipe Inspector Service
After=network.target

[Service]
Type=forking
User=ppak
WorkingDirectory=/home/ppak/pipe-inspector-electron
ExecStart=/home/ppak/pipe-inspector-electron/start.sh
ExecStop=/home/ppak/pipe-inspector-electron/stop.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**2. 서비스 활성화**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable pipe-inspector
sudo systemctl start pipe-inspector
sudo systemctl status pipe-inspector
```

---

## 📊 성능 모니터링

### 실시간 통계 확인
```bash
# 1초마다 상태 확인
watch -n 1 './status.sh'
```

### CPU/메모리 사용량 추적
```bash
# 프로세스별 리소스 사용량
ps aux | grep -E "python.*(backend_proxy|api\.py)" | grep -v grep
```

### GPU 메모리 모니터링
```bash
# GPU 사용량 실시간 확인
watch -n 1 nvidia-smi
```

---

## 🚦 서비스 상태 코드

### 정상 상태
- ✅ Port 5001 available
- ✅ Port 5002 available
- ✅ Backend Proxy running
- ✅ GPU Server running

### 부분 실행 상태
- ⚠️ Only Backend running
- ⚠️ Only GPU Server running

### 에러 상태
- ❌ Port already in use
- ❌ Failed to start
- ❌ Process not responding

---

## 💡 팁

### 빠른 재시작
```bash
# 한 줄로 중지 후 시작
./stop.sh && sleep 1 && ./start.sh

# 또는 재시작 스크립트 사용 (권장)
./restart.sh
```

### 로그 정리
```bash
# 오래된 로그 삭제
rm -f backend-proxy.log gpu-server.log

# 또는 백업 후 삭제
mv backend-proxy.log backend-proxy.log.backup
mv gpu-server.log gpu-server.log.backup
```

### 디버그 모드
```bash
# 수동으로 foreground에서 실행
python3 backend_proxy.py
# 또는
cd gpu-server && python3 api.py
```

---

## 📞 문의 및 지원

### 로그 확인 순서
1. `./status.sh` - 전체 상태 확인
2. `tail -f backend-proxy.log` - 백엔드 로그 확인
3. `tail -f gpu-server.log` - GPU 서버 로그 확인

### 일반적인 해결 순서
1. `./stop.sh` - 모든 서비스 중지
2. 로그 파일 확인
3. `./start.sh` - 서비스 재시작
4. `./status.sh` - 상태 확인

### 긴급 상황
```bash
# 모든 프로세스 강제 종료
pkill -9 -f "python.*backend_proxy"
pkill -9 -f "python.*api\.py"

# 포트 강제 해제
lsof -ti:5001,5002 | xargs kill -9

# 재시작
./restart.sh
```

---

## ✅ 체크리스트

**시작 전 확인**:
- [ ] Python 3 설치 확인
- [ ] 필요한 패키지 설치 확인
- [ ] GPU 드라이버 확인 (nvidia-smi)
- [ ] 포트 5001, 5002 사용 가능

**정상 작동 확인**:
- [ ] `./status.sh` 실행 시 모든 서비스 Running
- [ ] http://localhost:5001 접속 가능
- [ ] 로그인 화면 표시
- [ ] admin/admin123으로 로그인 성공

**종료 전 확인**:
- [ ] 진행 중인 작업 저장
- [ ] `./stop.sh` 실행
- [ ] 모든 프로세스 종료 확인

---

편리한 서버 관리를 위해 이 스크립트들을 활용하세요! 🚀
