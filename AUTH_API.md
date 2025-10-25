# 사용자 인증 API 문서

## 개요

Backend Proxy에 사용자 인증 및 세션 관리 시스템이 추가되었습니다.

## 구현된 기능

### 1. 사용자 관리
- ✅ 비밀번호 해싱 (SHA256)
- ✅ 세션 관리 (8시간 타임아웃)
- ✅ 자동 세션 갱신
- ✅ 주기적 세션 정리 (10분마다)
- ✅ 사용자별 프로젝트 폴더 격리

### 2. 기본 계정
```
User ID: admin
Password: admin123
Role: admin
```

## API 엔드포인트

### 인증 API

#### 1. 로그인
```http
POST /api/auth/login
Content-Type: application/json

{
  "user_id": "admin",
  "password": "admin123"
}
```

**응답 (성공)**:
```json
{
  "success": true,
  "session_id": "abc123...",
  "user": {
    "user_id": "admin",
    "full_name": "Administrator",
    "created_at": "2025-01-01T00:00:00",
    "role": "admin",
    "projects_dir": "projects/admin"
  }
}
```

**응답 (실패)**:
```json
{
  "success": false,
  "error": "Invalid user ID or password"
}
```

#### 2. 로그아웃
```http
POST /api/auth/logout
X-Session-ID: {session_id}
```

**응답**:
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

#### 3. 현재 사용자 정보 조회
```http
GET /api/auth/me
X-Session-ID: {session_id}
```

**응답**:
```json
{
  "success": true,
  "user": {
    "user_id": "admin",
    "full_name": "Administrator",
    "role": "admin",
    "projects_dir": "projects/admin"
  }
}
```

### 관리자 API (admin 권한 필요)

#### 1. 사용자 목록 조회
```http
GET /api/auth/users
X-Session-ID: {admin_session_id}
```

**응답**:
```json
{
  "success": true,
  "users": [
    {
      "user_id": "admin",
      "full_name": "Administrator",
      "role": "admin"
    },
    {
      "user_id": "user1",
      "full_name": "User One",
      "role": "user"
    }
  ]
}
```

#### 2. 새 사용자 생성
```http
POST /api/auth/users
X-Session-ID: {admin_session_id}
Content-Type: application/json

{
  "user_id": "user1",
  "password": "password123",
  "full_name": "User One",
  "role": "user"
}
```

**응답**:
```json
{
  "success": true,
  "message": "User created successfully",
  "user_id": "user1"
}
```

### 프로젝트 API (인증 필요)

모든 프로젝트 관련 API는 이제 인증이 필요하며, 사용자별로 격리됩니다.

#### 1. 프로젝트 목록 조회
```http
GET /api/projects
X-Session-ID: {session_id}
```

프로젝트는 `projects/{user_id}/` 디렉토리에 저장됩니다.

#### 2. 프로젝트 생성
```http
POST /api/projects
X-Session-ID: {session_id}
Content-Type: application/json

{
  "name": "My Project"
}
```

#### 3. 어노테이션 저장
```http
POST /api/projects/{project_id}/videos/{video_id}/annotations
X-Session-ID: {session_id}
Content-Type: application/json

{
  "annotations": {...}
}
```

## 세션 관리

### 세션 ID 전달 방법

두 가지 방법으로 세션 ID를 전달할 수 있습니다:

1. **HTTP 헤더** (권장):
```http
X-Session-ID: {session_id}
```

2. **쿠키**:
```http
Cookie: session_id={session_id}
```

### 세션 타임아웃

- **기본 타임아웃**: 8시간
- **자동 갱신**: API 요청 시 자동으로 8시간 연장
- **자동 정리**: 만료된 세션은 10분마다 자동 삭제

## 에러 코드

### 인증 에러

```json
{
  "success": false,
  "error": "Authentication required",
  "code": "NO_SESSION"
}
```
상태 코드: `401 Unauthorized`

```json
{
  "success": false,
  "error": "Invalid or expired session",
  "code": "INVALID_SESSION"
}
```
상태 코드: `401 Unauthorized`

### 권한 에러

```json
{
  "success": false,
  "error": "Admin privilege required"
}
```
상태 코드: `403 Forbidden`

```json
{
  "success": false,
  "error": "Access denied"
}
```
상태 코드: `403 Forbidden`

## 파일 구조

```
pipe-inspector-electron/
├── user_manager.py          # 사용자 관리 모듈
├── backend_proxy.py          # 인증이 통합된 백엔드 프록시
├── users.json                # 사용자 데이터 (자동 생성)
└── projects/                 # 프로젝트 루트 디렉토리
    ├── admin/                # 관리자 프로젝트
    │   └── project1/
    │       ├── project.json
    │       └── annotations/
    └── user1/                # 일반 사용자 프로젝트
        └── project2/
            ├── project.json
            └── annotations/
```

## 보안 고려사항

1. **비밀번호 해싱**: SHA256을 사용하여 비밀번호를 해싱
2. **세션 토큰**: `secrets.token_urlsafe(32)`를 사용한 안전한 토큰 생성
3. **프로젝트 격리**: 사용자는 자신의 프로젝트만 접근 가능
4. **권한 검증**: 관리자 전용 API는 role 확인
5. **세션 만료**: 8시간 후 자동 만료 및 정리

## 테스트 방법

### 1. 로그인 테스트
```bash
curl -X POST http://localhost:5001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_id":"admin","password":"admin123"}'
```

### 2. 현재 사용자 조회
```bash
curl http://localhost:5001/api/auth/me \
  -H "X-Session-ID: {session_id}"
```

### 3. 프로젝트 생성
```bash
curl -X POST http://localhost:5001/api/projects \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: {session_id}" \
  -d '{"name":"Test Project"}'
```

### 4. 새 사용자 생성 (관리자만)
```bash
curl -X POST http://localhost:5001/api/auth/users \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: {admin_session_id}" \
  -d '{"user_id":"testuser","password":"test123","full_name":"Test User"}'
```

## 다음 단계

프론트엔드에 로그인 UI를 추가하여 사용자 인증 시스템을 완성해야 합니다:

1. 로그인 페이지 생성
2. 세션 ID 저장 (localStorage 또는 쿠키)
3. API 요청 시 세션 ID 자동 전달
4. 세션 만료 시 자동 로그인 페이지 이동
5. 사용자 정보 표시 (현재 로그인 사용자)
