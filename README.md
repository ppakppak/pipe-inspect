# 🔧 Pipe Video Inspector - Electron

AI 기반 파이프 비디오 검사 및 어노테이션 시스템

---

## ✨ 주요 기능

### 🎯 핵심 기능
- **비디오 어노테이션**: 프레임별 녹(rust)과 스케일(scale) 영역 표시
- **AI 자동 검출**: SegFormer 모델을 사용한 자동 세그멘테이션
- **멀티유저 지원**: 사용자별 프로젝트 격리 및 권한 관리
- **실시간 추론**: GPU 가속 AI 추론

### 🔐 인증 시스템
- 사용자 로그인/로그아웃
- 세션 관리 (8시간 타임아웃)
- 사용자별 프로젝트 격리
- 관리자 권한 관리

### 🚀 성능
- **동시 사용자**: 5-10명
- **GPU 메모리**: 1.5GB (공유 모델)
- **멀티스레드**: 동시 요청 처리

---

## 📋 시스템 요구사항

### 필수 요구사항
- Python 3.8+
- NVIDIA GPU (CUDA 지원)
- 24GB GPU 메모리 권장
- 16GB 시스템 메모리 권장

### Python 패키지
```bash
pip install flask flask-cors torch torchvision transformers pillow opencv-python numpy
```

---

## 🚀 빠른 시작

### 1. 서버 시작

**방법 1: 스크립트 사용 (권장)**
```bash
cd /home/ppak/pipe-inspector-electron
./start.sh
```

**방법 2: 수동 실행**
```bash
# GPU 서버 (터미널 1)
cd gpu-server
python3 api.py

# Backend Proxy (터미널 2)
cd /home/ppak/pipe-inspector-electron
python3 backend_proxy.py
```

### 2. 브라우저 접속
```
http://localhost:5001
```

### 3. 로그인
```
사용자 ID: admin
비밀번호: admin123
```

---

## 🛠️ 서버 관리 스크립트

### 기본 스크립트

| 스크립트 | 설명 | 사용법 |
|---------|------|--------|
| `start.sh` | 서버 시작 | `./start.sh` |
| `stop.sh` | 서버 중지 | `./stop.sh` |
| `restart.sh` | 서버 재시작 (강제) | `./restart.sh` |
| `status.sh` | 상태 확인 | `./status.sh` |

### 사용 예시

**서버 시작**:
```bash
./start.sh
```

**상태 확인**:
```bash
./status.sh
```

**서버 재시작**:
```bash
./restart.sh
```

**서버 중지**:
```bash
./stop.sh
```

자세한 내용은 [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md)를 참조하세요.

---

## 📁 프로젝트 구조

```
pipe-inspector-electron/
├── index.html              # 프론트엔드 (로그인 UI 포함)
├── backend_proxy.py        # 백엔드 프록시 (인증 통합)
├── user_manager.py         # 사용자 관리 모듈
├── users.json              # 사용자 데이터 (자동 생성)
│
├── gpu-server/
│   └── api.py              # GPU 서버 (멀티스레드)
│
├── projects/               # 프로젝트 루트
│   ├── admin/              # 관리자 프로젝트
│   └── {user_id}/          # 사용자별 프로젝트
│
├── start.sh                # 서버 시작
├── stop.sh                 # 서버 중지
├── restart.sh              # 서버 재시작
├── status.sh               # 상태 확인
│
└── docs/                   # 문서
    ├── AUTH_API.md         # 인증 API 문서
    ├── LOGIN_GUIDE.md      # 로그인 가이드
    └── SCRIPTS_GUIDE.md    # 스크립트 가이드
```

---

## 🔌 API 엔드포인트

### 인증 API

| 메서드 | 엔드포인트 | 설명 | 인증 |
|--------|-----------|------|------|
| POST | `/api/auth/login` | 로그인 | ❌ |
| POST | `/api/auth/logout` | 로그아웃 | ✅ |
| GET | `/api/auth/me` | 현재 사용자 정보 | ✅ |
| GET | `/api/auth/users` | 사용자 목록 (관리자) | ✅ |
| POST | `/api/auth/users` | 사용자 생성 (관리자) | ✅ |

### 프로젝트 API (모두 인증 필요)

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| GET | `/api/projects` | 프로젝트 목록 (사용자별) |
| POST | `/api/projects` | 프로젝트 생성 |
| GET | `/api/projects/{id}` | 프로젝트 상세 |
| POST | `/api/projects/{id}/videos` | 비디오 추가 |
| GET | `/api/projects/{id}/videos/{vid}` | 비디오 정보 |

### AI API

| 메서드 | 엔드포인트 | 설명 |
|--------|-----------|------|
| POST | `/api/ai/initialize` | AI 모델 초기화 |
| POST | `/api/ai/inference` | 전체 프레임 추론 |
| POST | `/api/ai/inference_box` | 박스 영역 추론 |

자세한 API 문서는 [AUTH_API.md](AUTH_API.md)를 참조하세요.

---

## 👥 사용자 관리

### 기본 계정
```
사용자 ID: admin
비밀번호: admin123
역할: 관리자
```

### 새 사용자 생성 (관리자만)

**웹 브라우저 콘솔에서**:
```javascript
fetch('http://localhost:5001/api/auth/users', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-Session-ID': localStorage.getItem('session_id')
    },
    body: JSON.stringify({
        user_id: 'new_user',
        password: 'password123',
        full_name: 'New User',
        role: 'user'
    })
}).then(r => r.json()).then(console.log)
```

**curl 명령어로**:
```bash
# 1. admin으로 로그인
SESSION_ID=$(curl -s -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_id":"admin","password":"admin123"}' | jq -r '.session_id')

# 2. 새 사용자 생성
curl -X POST http://localhost:5001/api/auth/users \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: $SESSION_ID" \
  -d '{"user_id":"new_user","password":"password123","full_name":"New User"}'
```

---

## 🔍 로그 확인

### Backend Proxy 로그
```bash
tail -f /home/ppak/pipe-inspector-electron/backend-proxy.log
```

### GPU 서버 로그
```bash
tail -f /home/ppak/pipe-inspector-electron/gpu-server.log
```

### 실시간 모니터링
```bash
# 여러 터미널 창에서
./status.sh              # 터미널 1: 상태
tail -f backend-proxy.log  # 터미널 2: 백엔드
tail -f gpu-server.log     # 터미널 3: GPU
```

---

## 🛠️ 트러블슈팅

### 포트 충돌
```bash
# 포트 사용 중인 프로세스 확인
lsof -i:5001
lsof -i:5002

# 강제 재시작
./restart.sh
```

### 서버 시작 실패
```bash
# 로그 확인
tail -n 50 backend-proxy.log
tail -n 50 gpu-server.log

# 수동 실행으로 에러 확인
python3 backend_proxy.py
```

### GPU 메모리 부족
```bash
# GPU 상태 확인
nvidia-smi

# 서버 재시작
./restart.sh
```

### 세션 만료
- 8시간 후 자동 만료
- 다시 로그인 필요
- localStorage 확인: `localStorage.getItem('session_id')`

---

## 📊 성능 모니터링

### 실시간 상태 확인
```bash
watch -n 1 './status.sh'
```

### GPU 메모리 모니터링
```bash
watch -n 1 nvidia-smi
```

### 프로세스 리소스 확인
```bash
ps aux | grep -E "python.*(backend_proxy|api\.py)"
```

---

## 🔐 보안

### 비밀번호 보안
- SHA256 해싱
- 원본 비밀번호 저장 안 함

### 세션 보안
- `secrets.token_urlsafe(32)` 사용
- 8시간 타임아웃
- 자동 갱신 및 정리

### API 보안
- 모든 프로젝트 API 인증 필요
- 세션 ID 검증
- 401 Unauthorized 자동 처리

---

## 📈 성능 지표

### 현재 성능
- **동시 사용자**: 5-10명
- **GPU 메모리**: 1.5GB (공유 모델)
- **응답 시간**: <200ms (API)
- **세션 타임아웃**: 8시간

### 확장 계획
1. Redis 세션 저장
2. Gunicorn/uWSGI 프로덕션 서버
3. 다중 GPU 서버 로드 밸런싱
4. HTTPS/SSL 인증서

---

## 📚 문서

| 문서 | 설명 |
|------|------|
| [AUTH_API.md](AUTH_API.md) | 인증 API 상세 문서 |
| [LOGIN_GUIDE.md](LOGIN_GUIDE.md) | 로그인 시스템 가이드 |
| [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) | 서버 관리 스크립트 가이드 |

---

## 🎯 다음 단계

1. **프로젝트 생성**: 로그인 후 새 프로젝트 생성
2. **비디오 추가**: 프로젝트에 비디오 업로드
3. **AI 초기화**: GPU 서버에 AI 모델 로드
4. **어노테이션 시작**: 프레임별 영역 표시 및 AI 추론

---

## 📞 지원

### 문제 해결 순서
1. `./status.sh` - 전체 상태 확인
2. 로그 파일 확인
3. `./restart.sh` - 서버 재시작
4. [SCRIPTS_GUIDE.md](SCRIPTS_GUIDE.md) 참조

### 긴급 상황
```bash
# 모든 프로세스 강제 종료
pkill -9 -f "python.*backend_proxy"
pkill -9 -f "python.*api\.py"

# 재시작
./restart.sh
```

---

## 📝 라이선스

이 프로젝트는 내부 사용을 위한 것입니다.

---

**Happy Inspecting! 🚀**
