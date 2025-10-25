# 재시작 스크립트 사용 가이드

## 🏗️ 프로젝트 구조

이 프로젝트는 **Backend (Python Flask)** + **Frontend (Electron)** 구조입니다:
- **Backend**: `backend.py` - Flask API 서버 (포트 5003)
- **Frontend**: Electron 앱 - UI 인터페이스

## 📋 스크립트 목록

### 1️⃣ 전체 앱 시작 (권장)
```bash
npm run start:all
# 또는
./scripts/start-all.sh
```
- Backend (Python Flask) + Frontend (Electron) 모두 시작
- **가장 많이 사용할 명령어**

### 2️⃣ 개별 시작
```bash
# Backend만 시작
npm run start:backend

# Frontend만 시작
npm run start:frontend
```

### 3️⃣ 전체 종료
```bash
npm run stop
# 또는
./scripts/stop-all.sh
```

### 4️⃣ 재시작

**전체 재시작 (Backend + Frontend)**
```bash
npm run restart
```

**Backend만 재시작 (backend.py 변경 시)**
```bash
npm run restart:backend
```

**Electron Main Process만 재시작 (main.js 변경 시)**
```bash
npm run restart:main
```

**Frontend Renderer만 재시작 (index.html, CSS 변경 시)**
```bash
npm run restart:frontend
```
- **필요 패키지**: `sudo apt install xdotool`

### 5️⃣ 개발 모드 (자동 재시작)
```bash
npm run dev
# 또는
./scripts/dev-watch.sh
```
- 파일 변경 자동 감지 및 재시작
- `backend.py` → Backend 재시작
- `main.js` → Electron Main Process 재시작
- `*.html, *.css` → Renderer 재로드
- **필요 패키지**: `sudo apt install inotify-tools xdotool`

## 🛠️ 필수 패키지 설치

```bash
# Ubuntu/Debian
sudo apt install inotify-tools xdotool

# 설치 확인
which inotifywait  # 파일 감시 도구
which xdotool      # 윈도우 제어 도구
```

## 💡 사용 팁

### 개발 중 권장 워크플로우

1. **일반 개발**: `npm run dev` (자동 재시작)
2. **수동 제어 필요 시**:
   - Frontend 변경 → `npm run restart:frontend` (빠름)
   - Backend 변경 → `npm run restart:backend`
   - 전체 재시작 → `npm run restart`

### 키보드 단축키 (앱 내부)

- `Ctrl + R`: Frontend 재로드
- `Ctrl + Shift + R`: 캐시 무시하고 재로드
- `Ctrl + Shift + I`: DevTools 열기

## 🔧 문제 해결

### xdotool 없이 Frontend 재시작
앱 창에서 수동으로 `Ctrl + R` 입력

### 프로세스가 종료 안 될 때
```bash
pkill -9 -f electron
```

### 포트 충돌 시
```bash
# Electron 프로세스 확인
ps aux | grep electron
# 종료
kill -9 <PID>
```
