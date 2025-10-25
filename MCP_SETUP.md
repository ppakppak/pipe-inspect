# MCP Server 구성 가이드

Pipe Inspector Electron 앱의 MCP 서버 아키텍처 문서

## 🏗️ 아키텍처 개요

```
┌─────────────────────────┐
│  클라이언트 PC (No GPU)  │
│                         │
│  ┌──────────────────┐   │
│  │ Electron App (UI)│   │
│  └────────┬─────────┘   │
│           │ HTTP        │
│  ┌────────▼─────────┐   │
│  │ Flask Backend    │   │
│  │ (MCP Client)     │   │
│  └────────┬─────────┘   │
│           │ stdio       │
└───────────┼─────────────┘
            │
            │ MCP Protocol
            │
┌───────────▼─────────────┐
│    GPU 서버              │
│                         │
│  ┌──────────────────┐   │
│  │   MCP Server     │   │
│  │  (Python 3.11)   │   │
│  └────────┬─────────┘   │
│           │             │
│  ┌────────▼─────────┐   │
│  │  Grounded-SAM    │   │
│  │  (GPU Tasks)     │   │
│  └──────────────────┘   │
└─────────────────────────┘
```

## 📦 환경 구성

### 1. Python 환경

**Conda 환경: `mcp-server`**
- Python 3.11
- 패키지:
  - `mcp>=1.16.0` - MCP SDK
  - `flask>=3.0.0` - Flask 웹 프레임워크
  - `flask-cors` - CORS 지원
  - `opencv-python` - 비디오 처리
  - `numpy` - 수치 계산

### 2. 프로젝트 구조

```
pipe-inspector-electron/
├── backend.py              # Flask Backend (MCP Client)
├── backend_local.py        # 백업: 기존 로컬 버전
├── mcp_client.py          # MCP Client 래퍼
├── config.json            # 설정 파일
├── requirements.txt       # Python 의존성
├── mcp-server/
│   ├── server.py         # MCP Server (GPU 서버용)
│   └── requirements.txt  # MCP Server 의존성
└── scripts/
    ├── setup-dependencies.sh    # 의존성 설치
    ├── start-backend.sh         # Backend 시작
    ├── start-mcp-server.sh      # MCP Server 시작
    ├── test-mcp-server.sh       # MCP Server 테스트
    └── start-all.sh             # 전체 시작
```

## 🚀 설치 및 실행

### 1단계: Python 환경 준비

```bash
# Conda 환경 생성 (이미 완료됨)
conda create -n mcp-server python=3.11 -y

# 의존성 설치
./scripts/setup-dependencies.sh
```

### 2단계: 모드 선택

#### A. 로컬 모드 (GPU 서버에서 직접 실행)

```bash
# 전체 앱 시작 (Backend + Frontend)
npm run start:all
```

- Backend가 MCP Server를 subprocess로 실행
- 같은 머신에서 모든 것이 실행됨

#### B. 원격 모드 (클라이언트 PC → GPU 서버)

**GPU 서버에서:**
```bash
# MCP Server만 실행
./scripts/start-mcp-server.sh
```

**클라이언트 PC에서:**
```bash
# config.json에서 mode를 "remote"로 변경
# Backend + Frontend 실행
npm run start:all
```

## 🧪 테스트

### MCP Server 테스트

```bash
./scripts/test-mcp-server.sh
```

출력 예시:
```
🧪 Testing MCP Server...

📡 Connecting to MCP Server...
✅ Connected!

🔧 Listing available tools:
  - create_project: 새로운 프로젝트 생성
  - list_projects: 모든 프로젝트 목록 조회
  - get_project: 프로젝트 상세 정보 조회
  - add_video: 프로젝트에 비디오 추가
  ...

✅ Test complete
```

### API 테스트

```bash
# Backend 헬스 체크
curl http://localhost:5001/api/health

# MCP 도구 목록 조회
curl http://localhost:5001/api/mcp/tools

# 프로젝트 목록 조회
curl http://localhost:5001/api/projects
```

## 🔧 MCP 도구 (Tools)

MCP Server가 제공하는 도구들:

### 프로젝트 관리
- `create_project` - 프로젝트 생성
- `list_projects` - 프로젝트 목록
- `get_project` - 프로젝트 상세 정보
- `delete_project` - 프로젝트 삭제

### 비디오 관리
- `add_video` - 비디오 추가
- `remove_video` - 비디오 제거

### 통계
- `get_project_statistics` - 프로젝트 통계

## 📊 REST API 엔드포인트

Flask Backend가 제공하는 HTTP API:

```
GET  /api/health                              - 헬스 체크
GET  /api/test                                - 테스트
GET  /api/mcp/tools                           - MCP 도구 목록

GET  /api/projects                            - 프로젝트 목록
POST /api/projects                            - 프로젝트 생성
GET  /api/projects/<id>                       - 프로젝트 조회
GET  /api/projects/<id>/statistics            - 통계 조회

POST   /api/projects/<id>/videos              - 비디오 추가
DELETE /api/projects/<id>/videos/<video_id>   - 비디오 제거
```

## 🐛 트러블슈팅

### 1. ModuleNotFoundError: No module named 'mcp'

**원인**: mcp-server conda 환경이 활성화되지 않음

**해결**:
```bash
conda run -n mcp-server pip install mcp
```

### 2. MCP Server 연결 실패

**확인 사항**:
- MCP Server가 실행 중인지 확인
- `mcp-server/server.py` 경로가 올바른지 확인
- 로그 확인: `tail -f backend.log`

### 3. Grounded-SAM 경로 오류

**해결**: `config.json`에서 `grounded_sam.project_base_dir` 경로 확인

## 🔄 마이그레이션 가이드

### 기존 로컬 버전에서 MCP 버전으로

1. **백업 보관**: `backend_local.py` (기존 버전)
2. **새 버전 사용**: `backend.py` (MCP Client)
3. **전환 방법**:
   ```bash
   # MCP 버전 사용
   npm run start:all

   # 기존 로컬 버전으로 롤백하려면
   mv backend.py backend_mcp.py
   mv backend_local.py backend.py
   ```

## 📝 설정 파일 (config.json)

```json
{
  "mode": "local",
  "local": {
    "mcp_server_script": "./mcp-server/server.py",
    "backend_port": 5001
  },
  "remote": {
    "mcp_server_host": "192.168.0.100",
    "mcp_server_port": 5002,
    "backend_port": 5001
  }
}
```

## 🎯 다음 단계

1. **GPU 작업 추가**: Grounded-SAM 추론 도구 구현
2. **원격 모드**: SSE 또는 WebSocket으로 원격 MCP 서버 지원
3. **배치 처리**: 여러 비디오 동시 처리
4. **캐싱**: 결과 캐싱으로 성능 향상

## 📚 참고 자료

- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Flask Documentation](https://flask.palletsprojects.com/)
