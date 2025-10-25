#!/bin/bash

# Pipe Inspector 실행 관리 스크립트
# - config.json의 mode를 읽어 적절한 백엔드/프론트엔드를 시작
# - --mode 옵션으로 수동 전환 가능 (local | remote | gpu)
# - --stop 으로 모든 관련 프로세스 종료

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config.json"
BACKEND_LOG="${PROJECT_ROOT}/backend.log"
GPU_LOG="${PROJECT_ROOT}/gpu-server.log"

MODE="auto"
ACTION="start"
RUN_FRONTEND="yes"
RUN_BACKEND="yes"

usage() {
  cat <<'EOF'
사용법: scripts/run-project.sh [옵션]

옵션:
  --mode <local|remote|gpu>  실행 모드 강제 지정 (기본 auto: config.json 사용)
  --backend-only             백엔드만 실행
  --frontend-only            프론트엔드만 실행
  --stop                     실행 중인 프로세스 종료
  --status                   프로세스 상태 확인
  -h, --help                 도움말 표시

모드 설명:
  local   : Quart + MCP 백엔드와 Electron 프론트엔드 실행
  remote  : backend_proxy.py (GPU 서버로 프록시) + Electron 실행
  gpu     : GPU 서버용 REST API (gpu-server/api.py) 실행
EOF
}

log() {
  echo "[run-project] $*"
}

detect_mode() {
  if [[ "$MODE" != "auto" ]]; then
    echo "$MODE"
    return
  fi

  if [[ ! -f "$CONFIG_FILE" ]]; then
    log "config.json을 찾지 못해 기본(local) 모드로 실행합니다."
    echo "local"
    return
  fi

  python3 - "$CONFIG_FILE" <<'PY' || echo "local"
import json, sys
cfg_path = sys.argv[1]
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    mode = data.get("mode", "local").strip()
    if mode in {"local", "remote", "gpu"}:
        print(mode)
    else:
        print("local")
except Exception:
    print("local")
PY
}

ensure_mode_supported() {
  case "$1" in
    local|remote|gpu) ;;
    *)
      log "지원하지 않는 모드입니다: $1"
      usage
      exit 1
      ;;
  esac
}

stop_process() {
  local pattern="$1"
  local label="$2"
  if pkill -f "$pattern" 2>/dev/null; then
    log "$label 중지 완료"
  else
    log "$label 실행 중이 아님"
  fi
}

show_status() {
  printf "%-25s %s\n" "프로세스" "상태"
  printf "%-25s %s\n" "-------------------------" "---------"
  if pgrep -f "python.*backend.py" >/dev/null; then
    printf "%-25s %s\n" "backend.py" "RUNNING"
  else
    printf "%-25s %s\n" "backend.py" "STOPPED"
  fi
  if pgrep -f "python.*backend_proxy.py" >/dev/null; then
    printf "%-25s %s\n" "backend_proxy.py" "RUNNING"
  else
    printf "%-25s %s\n" "backend_proxy.py" "STOPPED"
  fi
  if pgrep -f "python.*gpu-server/api.py" >/dev/null; then
    printf "%-25s %s\n" "gpu-server/api.py" "RUNNING"
  else
    printf "%-25s %s\n" "gpu-server/api.py" "STOPPED"
  fi
  if pgrep -f "electron.*pipe-inspector-electron" >/dev/null; then
    printf "%-25s %s\n" "Electron" "RUNNING"
  else
    printf "%-25s %s\n" "Electron" "STOPPED"
  fi
}

start_local_backend() {
  log "로컬 MCP 백엔드를 시작합니다."
  bash "${SCRIPT_DIR}/start-backend.sh"
}

start_remote_backend() {
  log "원격 프록시 백엔드를 시작합니다."
  local gpu_info
  mapfile -t gpu_info < <(python3 - "$CONFIG_FILE" <<'PY'
import json, sys
cfg_path = sys.argv[1]
try:
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    remote = data.get("remote", {})
    host = remote.get("mcp_server_host", "localhost")
    port = remote.get("mcp_server_port", 5004)
    print(f"http://{host}:{port}")
    print(host)
except Exception:
    print("http://localhost:5004")
    print("localhost")
PY
)
  local gpu_url="${gpu_info[0]}"
  local gpu_host="${gpu_info[1]:-localhost}"
  log "GPU 서버 URL: $gpu_url"

  if [[ "$gpu_host" =~ ^(localhost|127\.0\.0\.1)$ ]]; then
    log "로컬 GPU 서버 감지 - gpu-server/api.py를 시작합니다."
    start_gpu_server
  else
    log "원격 GPU 서버(${gpu_host}) - 로컬 GPU 서버는 시작하지 않습니다."
  fi

  stop_process "python.*backend_proxy.py" "기존 backend_proxy.py"

  (
    cd "$PROJECT_ROOT"
    export GPU_SERVER_URL="$gpu_url"
    nohup python3 backend_proxy.py >>"$BACKEND_LOG" 2>&1 &
    log "backend_proxy.py PID: $!"
  )
}

start_gpu_server() {
  log "GPU 서버 REST API를 시작합니다."
  stop_process "python.*gpu-server/api.py" "기존 gpu-server/api.py"

  # GPU 서버에 필요한 conda 환경 선택 (torch와 flask가 모두 설치된 환경)
  # dino 환경을 최우선으로 사용
  local gpu_python=""
  for env in dino label_env labeler pipe_env; do
    local python_path="/home/ppak/.conda/envs/${env}/bin/python"
    if [[ -x "$python_path" ]]; then
      if $python_path -c "import torch, flask, transformers" 2>/dev/null; then
        gpu_python="$python_path"
        log "GPU 환경: $env (torch, flask, transformers 사용 가능)"
        break
      fi
    fi
  done

  if [[ -z "$gpu_python" ]]; then
    log "경고: torch가 설치된 conda 환경을 찾지 못했습니다. 기본 python3 사용"
    gpu_python="python3"
  fi

  (
    cd "$PROJECT_ROOT"
    nohup "$gpu_python" gpu-server/api.py >>"$GPU_LOG" 2>&1 &
    log "gpu-server/api.py PID: $!"
  )
}

start_frontend() {
  log "Electron 프론트엔드를 시작합니다."
  bash "${SCRIPT_DIR}/start-frontend.sh"
}

stop_all() {
  log "Pipe Inspector 관련 프로세스를 종료합니다."
  stop_process "python.*backend.py" "backend.py"
  stop_process "python.*backend_proxy.py" "backend_proxy.py"
  stop_process "python.*gpu-server/api.py" "gpu-server/api.py"
  stop_process "electron.*pipe-inspector-electron" "Electron"
  log "종료 작업이 완료되었습니다."
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ $# -ge 2 ]] || { log "--mode 옵션에는 값이 필요합니다."; exit 1; }
      MODE="$2"
      shift 2
      ;;
    --backend-only)
      RUN_FRONTEND="no"
      shift
      ;;
    --frontend-only)
      RUN_BACKEND="no"
      shift
      ;;
    --stop)
      ACTION="stop"
      shift
      ;;
    --status)
      ACTION="status"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log "알 수 없는 옵션: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "$RUN_BACKEND" == "no" && "$RUN_FRONTEND" == "no" ]]; then
  log "백엔드와 프론트엔드를 모두 비활성화할 수 없습니다."
  exit 1
fi

case "$ACTION" in
  stop)
    stop_all
    exit 0
    ;;
  status)
    show_status
    exit 0
    ;;
  start)
    ;;
  *)
    log "알 수 없는 동작: $ACTION"
    exit 1
    ;;
esac

EFFECTIVE_MODE="$(detect_mode)"
ensure_mode_supported "$EFFECTIVE_MODE"
log "실행 모드: $EFFECTIVE_MODE"

case "$EFFECTIVE_MODE" in
  local)
    if [[ "$RUN_BACKEND" == "yes" ]]; then
      start_local_backend
    fi
    ;;
  remote)
    if [[ "$RUN_BACKEND" == "yes" ]]; then
      start_remote_backend
    fi
    ;;
  gpu)
    if [[ "$RUN_BACKEND" == "yes" ]]; then
      start_gpu_server
    fi
    ;;
esac

if [[ "$RUN_FRONTEND" == "yes" && "$EFFECTIVE_MODE" != "gpu" ]]; then
  start_frontend
elif [[ "$RUN_FRONTEND" == "yes" && "$EFFECTIVE_MODE" == "gpu" ]]; then
  log "gpu 모드에서는 프론트엔드를 자동 실행하지 않습니다."
fi

log "실행 요청이 완료되었습니다."
log "백엔드 로그: $BACKEND_LOG"
log "GPU 로그:    $GPU_LOG"
