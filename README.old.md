# Pipe Inspector - Electron Edition

PyQt5에서 Electron으로 마이그레이션한 파이프 검사 시스템입니다.

## 🏗️ 아키텍처

- **Frontend**: Electron (HTML/CSS/JavaScript)
- **Backend**: Python Flask REST API
- **기존 코드**: PyQt5 프로젝트의 로직 재사용

## 🚀 시작하기

### 1. 백엔드 서버 실행

```bash
cd ~/pipe-inspector-electron
python backend.py
```

백엔드가 http://localhost:5000 에서 실행됩니다.

### 2. Electron 앱 실행

**새 터미널에서:**

```bash
cd ~/pipe-inspector-electron
npm start
```

## 📁 프로젝트 구조

```
pipe-inspector-electron/
├── main.js              # Electron 메인 프로세스
├── index.html           # UI (프론트엔드)
├── backend.py           # Flask API 서버
├── package.json         # npm 설정
└── README.md           # 이 파일
```

## 🔌 API 엔드포인트

### 헬스 체크
```
GET /api/health
```

### 프로젝트 목록
```
GET /api/projects
```

### 프로젝트 생성
```
POST /api/projects
Body: {
  "name": "프로젝트명",
  "classes": "scale,rust"
}
```

### 테스트
```
GET /api/test
```

## 📋 마이그레이션 진행 상황

- [x] Electron 기본 구조
- [x] Flask 백엔드 API
- [x] 프론트엔드-백엔드 통신
- [ ] 프로젝트 관리 기능
- [ ] 비디오 주석 기능
- [ ] 데이터셋 빌더
- [ ] 모델 학습
- [ ] 비디오 추론

## 🛠️ 개발 가이드

### 새 API 추가하기

1. `backend.py`에 Flask 라우트 추가
2. `index.html`에서 fetch로 호출
3. UI 업데이트

### 기존 PyQt5 코드 재사용

기존 프로젝트의 로직을 import해서 사용:

```python
# backend.py
from project_manager import ProjectManager
from pipe_inspector import SAMSegmenter, PipeDefectClassifier
```

## 🐛 트러블슈팅

### Electron 창이 안 뜰 때
```bash
# --no-sandbox 옵션 사용
npm start
```

### 백엔드 연결 오류
- backend.py가 실행 중인지 확인
- 포트 5000이 사용 가능한지 확인
- CORS 설정 확인

## 📝 다음 단계

1. Project Manager 페이지 완성
2. Video Player 컴포넌트 추가
3. SAM + CNN 통합
4. 데이터셋 빌더 UI
5. 학습/추론 인터페이스
