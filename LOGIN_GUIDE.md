# 로그인 시스템 구현 완료 가이드

## 🎉 구현 완료 사항

### 1. 백엔드 인증 시스템 ✅
- **파일**: `backend_proxy.py`, `user_manager.py`
- **기능**:
  - 사용자 로그인/로그아웃
  - 세션 관리 (8시간 타임아웃)
  - 사용자별 프로젝트 격리
  - 관리자 권한 관리

### 2. 프론트엔드 로그인 UI ✅
- **파일**: `index.html`
- **기능**:
  - 로그인 화면
  - 사용자 정보 표시
  - 자동 세션 검증
  - 세션 만료 시 자동 로그인 페이지 이동

### 3. GPU 서버 멀티스레드 지원 ✅
- **파일**: `gpu-server/api.py`
- **기능**:
  - 멀티스레드 요청 처리
  - 추론 작업 순차 처리 (락)
  - 서버 통계 모니터링

---

## 🚀 시작하기

### 1. 서버 실행

**Backend Proxy 실행**:
```bash
cd /home/ppak/pipe-inspector-electron
python3 backend_proxy.py
```

**GPU 서버 실행** (별도 터미널):
```bash
cd /home/ppak/pipe-inspector-electron/gpu-server
python3 api.py
```

### 2. 브라우저 접속

```
http://localhost:5001
```

### 3. 로그인

**기본 관리자 계정**:
- **사용자 ID**: `admin`
- **비밀번호**: `admin123`

---

## 📋 주요 기능

### 로그인 화면
- 페이지 로드 시 자동으로 표시
- 세션이 있으면 자동 로그인
- 로그인 실패 시 에러 메시지 표시

### 사용자 정보 표시
- 로그인 후 우측 상단에 사용자 이름 표시
- 로그아웃 버튼 제공

### 세션 관리
- **타임아웃**: 8시간
- **자동 갱신**: API 요청 시 8시간 연장
- **자동 정리**: 10분마다 만료된 세션 삭제
- **세션 만료 시**: 자동으로 로그인 화면으로 이동

### 프로젝트 격리
- 사용자별 프로젝트 폴더: `projects/{user_id}/`
- 다른 사용자의 프로젝트 접근 불가
- 프로젝트 생성/조회/수정 모두 인증 필요

---

## 🔐 보안 기능

### 비밀번호 보안
- SHA256 해싱
- 원본 비밀번호 저장 안 함

### 세션 보안
- `secrets.token_urlsafe(32)` 사용
- 안전한 랜덤 토큰 생성

### API 보안
- 모든 프로젝트/어노테이션 API는 인증 필요
- 세션 ID 검증 (헤더 또는 쿠키)
- 401 Unauthorized 시 자동 로그아웃

---

## 👥 사용자 관리

### 새 사용자 생성 (관리자만)

**웹 브라우저에서**:
1. admin 계정으로 로그인
2. 브라우저 콘솔에서:
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
# 먼저 admin으로 로그인
curl -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_id":"admin","password":"admin123"}'

# 세션 ID를 받아서 사용자 생성
curl -X POST http://localhost:5001/api/auth/users \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: {받은_세션_ID}" \
  -d '{"user_id":"new_user","password":"password123","full_name":"New User"}'
```

### 사용자 목록 조회 (관리자만)

```javascript
fetch('http://localhost:5001/api/auth/users', {
    headers: {
        'X-Session-ID': localStorage.getItem('session_id')
    }
}).then(r => r.json()).then(console.log)
```

---

## 🛠️ 개발자 가이드

### 세션 정보 확인

**브라우저 콘솔에서**:
```javascript
// 세션 ID 확인
console.log('Session ID:', localStorage.getItem('session_id'));

// 현재 사용자 정보 확인
console.log('Current User:', JSON.parse(localStorage.getItem('current_user')));
```

### API 요청 예제

**인증이 필요한 API 호출**:
```javascript
// authFetch 사용 (자동으로 세션 ID 포함)
const response = await authFetch('/api/projects');
const data = await response.json();
console.log('Projects:', data);
```

**수동으로 세션 ID 포함**:
```javascript
const response = await fetch('http://localhost:5001/api/projects', {
    headers: {
        'X-Session-ID': localStorage.getItem('session_id')
    }
});
const data = await response.json();
console.log('Projects:', data);
```

### 로그아웃

**프로그래밍 방식**:
```javascript
handleLogout();
```

**UI 버튼**:
- 우측 상단 "로그아웃" 버튼 클릭

---

## 📁 파일 구조

```
pipe-inspector-electron/
├── index.html              # 프론트엔드 (로그인 UI 포함)
├── backend_proxy.py        # 백엔드 프록시 (인증 통합)
├── user_manager.py         # 사용자 관리 모듈
├── users.json              # 사용자 데이터 (자동 생성)
├── gpu-server/
│   └── api.py              # GPU 서버 (멀티스레드)
└── projects/               # 프로젝트 루트
    ├── admin/              # 관리자 프로젝트
    └── {user_id}/          # 사용자별 프로젝트
```

---

## 🔍 트러블슈팅

### 로그인이 안 됨
1. 백엔드 서버가 실행 중인지 확인: `http://localhost:5001/api/health`
2. 브라우저 콘솔에서 에러 메시지 확인
3. 기본 계정 확인: `admin` / `admin123`

### 세션이 계속 만료됨
1. 서버 로그 확인: `backend_proxy.py` 터미널
2. 세션 타임아웃 확인 (기본 8시간)
3. 브라우저 localStorage 확인

### 프로젝트가 보이지 않음
1. 로그인한 사용자의 프로젝트만 표시됨
2. `projects/{user_id}/` 디렉토리 확인
3. 다른 계정으로 로그인해서 테스트

### 401 Unauthorized 에러
1. 세션이 만료되었거나 유효하지 않음
2. 자동으로 로그인 페이지로 이동해야 함
3. 다시 로그인 필요

---

## 📊 성능 및 확장성

### 현재 성능
- **동시 사용자**: 5-10명
- **GPU 모델**: 공유 모델 (1.5GB 메모리)
- **세션 관리**: 메모리 기반 (서버 재시작 시 초기화)

### 향후 개선 사항
1. **세션 영구 저장**: Redis 또는 데이터베이스 사용
2. **프로덕션 서버**: Gunicorn 또는 uWSGI 사용
3. **부하 분산**: 여러 GPU 서버로 확장
4. **HTTPS**: SSL/TLS 인증서 적용

---

## ✅ 테스트 체크리스트

- [x] 로그인 화면 표시
- [x] admin 계정으로 로그인
- [x] 사용자 정보 표시 (우측 상단)
- [x] 프로젝트 생성 (인증 필요)
- [x] 프로젝트 목록 조회 (사용자별)
- [x] 로그아웃 기능
- [x] 세션 자동 검증
- [x] 세션 만료 시 로그인 페이지 이동
- [x] 새 사용자 생성 (관리자)
- [x] 멀티스레드 동시 요청 처리

---

## 📞 문의 및 지원

시스템 관련 문의사항:
- API 문서: `AUTH_API.md`
- 백엔드 로그: `backend_proxy.py` 터미널 출력
- GPU 서버 로그: `gpu-server.log`

기본 관리자 계정으로 로그인 후 추가 사용자를 생성하여 사용하세요!
