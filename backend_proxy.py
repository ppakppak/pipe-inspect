#!/usr/bin/env python3
"""
Client Backend (Proxy Mode)
클라이언트 PC에서 GPU 서버로 요청을 전달하는 프록시
"""

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from functools import wraps
import requests
import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path

# 사용자 관리 모듈 임포트
from user_manager import UserManager

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# GPU 서버 URL (환경 변수 또는 기본값)
GPU_SERVER_URL = os.getenv('GPU_SERVER_URL', 'http://localhost:5004')

# 프로젝트 기본 디렉토리 (고정 경로)
BASE_PROJECTS_DIR = Path('/home/intu/Nas2/k_water/pipe_inspector_data')

# 기본 디렉토리 생성
BASE_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# 사용자 관리자 초기화
user_manager = UserManager()


def require_auth(f):
    """세션 검증 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 세션 ID 가져오기 (헤더, 쿠키 또는 쿼리 파라미터)
        session_id = request.headers.get('X-Session-ID') or request.cookies.get('session_id') or request.args.get('token')

        if not session_id:
            logger.warning("[AUTH] No session ID provided")
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'code': 'NO_SESSION'
            }), 401

        # 세션 검증
        user_id = user_manager.validate_session(session_id)
        if not user_id:
            logger.warning(f"[AUTH] Invalid or expired session: {session_id[:8]}...")
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session',
                'code': 'INVALID_SESSION'
            }), 401

        # 사용자 ID를 request에 저장
        request.user_id = user_id
        request.session_id = session_id

        return f(*args, **kwargs)

    return decorated_function


def require_admin(f):
    """관리자 권한 검증 데코레이터"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 먼저 인증 확인 (헤더, 쿠키 또는 쿼리 파라미터)
        session_id = request.headers.get('X-Session-ID') or request.cookies.get('session_id') or request.args.get('token')

        if not session_id:
            logger.warning("[ADMIN] No session ID provided")
            return jsonify({
                'success': False,
                'error': 'Authentication required',
                'code': 'NO_SESSION'
            }), 401

        # 세션 검증
        user_id = user_manager.validate_session(session_id)
        if not user_id:
            logger.warning(f"[ADMIN] Invalid or expired session: {session_id[:8]}...")
            return jsonify({
                'success': False,
                'error': 'Invalid or expired session',
                'code': 'INVALID_SESSION'
            }), 401

        # 관리자 권한 확인
        user_info = user_manager.get_user_info(user_id)
        if not user_info or user_info.get('role') != 'admin':
            logger.warning(f"[ADMIN] User {user_id} is not admin (role: {user_info.get('role') if user_info else 'None'})")
            return jsonify({
                'success': False,
                'error': 'Admin access required',
                'code': 'FORBIDDEN'
            }), 403

        # 사용자 ID를 request에 저장
        request.user_id = user_id
        request.session_id = session_id

        return f(*args, **kwargs)

    return decorated_function


def find_project_dir(project_id: str, user_id: str):
    """
    프로젝트 디렉토리 찾기
    - 먼저 자신의 프로젝트 폴더에서 검색
    - 없으면 모든 사용자 폴더에서 검색 (공유 프로젝트 접근)
    - 프로젝트 디렉토리 이름은 "이름_project_id" 형식이므로 endswith로 검색

    Returns:
        Path or None
    """
    from pathlib import Path

    base_dir = Path(BASE_PROJECTS_DIR)

    logger.info(f"[PROJECT] Searching for project_id={project_id} for user={user_id}")

    # 1. 먼저 자신의 프로젝트에서 찾기
    user_dir = base_dir / user_id
    if user_dir.exists() and user_dir.is_dir():
        for project_dir in user_dir.iterdir():
            if project_dir.is_dir() and (project_dir.name.endswith(f"_{project_id}") or project_dir.name == project_id):
                logger.info(f"[PROJECT] Found project: {project_dir}")
                return project_dir

    # 2. 자신의 프로젝트에 없으면 다른 사용자 폴더에서 검색
    for other_user_dir in base_dir.iterdir():
        if not other_user_dir.is_dir():
            continue

        # 자신의 디렉토리는 이미 확인했음
        if other_user_dir.name == user_id:
            continue

        for project_dir in other_user_dir.iterdir():
            if project_dir.is_dir() and (project_dir.name.endswith(f"_{project_id}") or project_dir.name == project_id):
                logger.info(f"[PROJECT] User {user_id} accessing shared project {project_id} from user {other_user_dir.name}: {project_dir}")
                return project_dir

    logger.warning(f"[PROJECT] Project {project_id} not found for user {user_id}")
    return None


def forward_to_gpu(path, method='GET', **kwargs):
    """GPU 서버로 요청 전달"""
    url = f"{GPU_SERVER_URL}{path}"
    logger.info(f"Forwarding {method} request to {url}")

    try:
        params = kwargs.get('params', None)

        # 데이터셋 빌드는 시간이 오래 걸리므로 타임아웃을 길게 설정
        is_dataset_build = '/dataset/build' in path
        timeout = 600 if is_dataset_build else 30  # 10분 vs 30초

        if method == 'GET':
            response = requests.get(url, params=params, timeout=timeout)
        elif method == 'POST':
            # files가 있으면 multipart/form-data, 없으면 JSON
            if 'files' in kwargs:
                response = requests.post(url, files=kwargs.get('files'), params=params, timeout=timeout)
            else:
                response = requests.post(url, json=kwargs.get('json'), params=params, timeout=timeout)
        elif method == 'DELETE':
            response = requests.delete(url, params=params, timeout=timeout)
        else:
            logger.error(f"Unsupported HTTP method: {method}")
            return {'success': False, 'error': 'Unsupported method'}, 400

        logger.info(f"GPU server response: {response.status_code}")
        return response.json(), response.status_code

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot connect to GPU server: {e}")
        return {
            'success': False,
            'error': 'Cannot connect to GPU server',
            'gpu_server': GPU_SERVER_URL
        }, 503
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timeout: {e}")
        return {
            'success': False,
            'error': 'Request timeout'
        }, 504
    except Exception as e:
        logger.error(f"Error forwarding request: {e}")
        return {
            'success': False,
            'error': str(e)
        }, 500


@app.route('/api/auth/login', methods=['POST'])
def login():
    """사용자 로그인"""
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        password = data.get('password')

        if not user_id or not password:
            return jsonify({
                'success': False,
                'error': 'User ID and password are required'
            }), 400

        # 인증 시도
        session_id = user_manager.authenticate(user_id, password)

        if not session_id:
            logger.warning(f"[AUTH] Failed login attempt for user: {user_id}")
            return jsonify({
                'success': False,
                'error': 'Invalid user ID or password'
            }), 401

        # 사용자 정보 조회
        user_info = user_manager.get_user_info(user_id)

        logger.info(f"[AUTH] User logged in: {user_id}")

        return jsonify({
            'success': True,
            'session_id': session_id,
            'user': user_info
        })

    except Exception as e:
        logger.error(f"[AUTH] Login error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/logout', methods=['POST'])
@require_auth
def logout():
    """사용자 로그아웃"""
    try:
        user_manager.logout(request.session_id)
        logger.info(f"[AUTH] User logged out: {request.user_id}")

        return jsonify({
            'success': True,
            'message': 'Logged out successfully'
        })

    except Exception as e:
        logger.error(f"[AUTH] Logout error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/me', methods=['GET'])
@require_auth
def get_current_user():
    """현재 로그인한 사용자 정보 조회"""
    try:
        user_info = user_manager.get_user_info(request.user_id)

        return jsonify({
            'success': True,
            'user': user_info
        })

    except Exception as e:
        logger.error(f"[AUTH] Get user info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/users', methods=['GET'])
@require_auth
def list_users():
    """사용자 목록 조회 (관리자 전용)"""
    try:
        # 관리자 권한 확인
        user_info = user_manager.get_user_info(request.user_id)
        if user_info.get('role') != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin privilege required'
            }), 403

        users = user_manager.list_users()

        return jsonify({
            'success': True,
            'users': users
        })

    except Exception as e:
        logger.error(f"[AUTH] List users error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/users', methods=['POST'])
@require_auth
def create_user():
    """새 사용자 생성 (관리자 전용)"""
    try:
        # 관리자 권한 확인
        user_info = user_manager.get_user_info(request.user_id)
        if user_info.get('role') != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin privilege required'
            }), 403

        data = request.get_json()
        new_user_id = data.get('user_id')
        password = data.get('password')
        full_name = data.get('full_name', '')
        role = data.get('role', 'user')

        if not new_user_id or not password:
            return jsonify({
                'success': False,
                'error': 'User ID and password are required'
            }), 400

        # 사용자 생성
        success = user_manager.create_user(new_user_id, password, full_name, role)

        if not success:
            return jsonify({
                'success': False,
                'error': 'User already exists'
            }), 400

        logger.info(f"[AUTH] New user created: {new_user_id} by {request.user_id}")

        return jsonify({
            'success': True,
            'message': 'User created successfully',
            'user_id': new_user_id
        })

    except Exception as e:
        logger.error(f"[AUTH] Create user error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/users/<user_id>', methods=['PUT'])
@require_auth
def update_user(user_id):
    """사용자 정보 수정 (관리자 전용)"""
    import json
    import shutil

    try:
        # 관리자 권한 확인
        user_info = user_manager.get_user_info(request.user_id)
        if user_info.get('role') != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin privilege required'
            }), 403

        data = request.get_json()
        new_user_id = data.get('new_user_id')
        full_name = data.get('full_name')
        role = data.get('role')
        password = data.get('password')

        # 최소한 하나의 필드는 제공되어야 함
        if new_user_id is None and full_name is None and role is None and password is None:
            return jsonify({
                'success': False,
                'error': 'At least one field (new_user_id, full_name, role, or password) is required'
            }), 400

        # user_id 변경 시 프로젝트 디렉토리 처리
        old_user_dir = None
        new_user_dir = None

        if new_user_id and new_user_id != user_id:
            # 사용자 디렉토리 경로
            old_user_dir = Path(BASE_PROJECTS_DIR) / user_id
            new_user_dir = Path(BASE_PROJECTS_DIR) / new_user_id

            # 새 디렉토리가 이미 존재하는지 확인
            if new_user_dir.exists():
                return jsonify({
                    'success': False,
                    'error': 'New user ID already has a project directory'
                }), 400

        # 사용자 정보 업데이트 (user_manager가 세션도 업데이트함)
        success = user_manager.update_user(
            user_id,
            new_user_id=new_user_id,
            full_name=full_name,
            role=role,
            password=password
        )

        if not success:
            return jsonify({
                'success': False,
                'error': 'User not found, invalid role, or new user ID already exists'
            }), 404

        # user_id가 변경되었고 프로젝트 디렉토리가 있으면 처리
        if new_user_id and new_user_id != user_id and old_user_dir and old_user_dir.exists():
            try:
                # 프로젝트 디렉토리 이름 변경
                logger.info(f"[AUTH] Renaming user directory: {old_user_dir} -> {new_user_dir}")
                old_user_dir.rename(new_user_dir)

                # 모든 project.json 파일의 user_id 업데이트
                for project_dir in new_user_dir.iterdir():
                    if project_dir.is_dir():
                        project_file = project_dir / 'project.json'
                        if project_file.exists():
                            try:
                                with open(project_file, 'r', encoding='utf-8') as f:
                                    project_data = json.load(f)

                                # user_id 업데이트
                                project_data['user_id'] = new_user_id

                                with open(project_file, 'w', encoding='utf-8') as f:
                                    json.dump(project_data, f, indent=2, ensure_ascii=False)

                                logger.info(f"[AUTH] Updated project.json: {project_file}")
                            except Exception as e:
                                logger.error(f"[AUTH] Failed to update project.json {project_file}: {e}")

                logger.info(f"[AUTH] Successfully updated all project ownership for user {new_user_id}")

            except Exception as e:
                logger.error(f"[AUTH] Failed to rename user directory: {e}")
                # 디렉토리 이름 변경 실패는 경고만 하고 계속 진행
                # (사용자 정보는 이미 업데이트 되었으므로)

        # 응답에는 변경된 user_id 사용
        final_user_id = new_user_id if (new_user_id and new_user_id != user_id) else user_id

        logger.info(f"[AUTH] User updated: {user_id} -> {final_user_id} by {request.user_id}")

        return jsonify({
            'success': True,
            'message': 'User updated successfully',
            'user_id': final_user_id
        })

    except Exception as e:
        logger.error(f"[AUTH] Update user error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/auth/users/<user_id>', methods=['DELETE'])
@require_auth
def delete_user(user_id):
    """사용자 삭제 (관리자 전용)"""
    try:
        # 관리자 권한 확인
        user_info = user_manager.get_user_info(request.user_id)
        if user_info.get('role') != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin privilege required'
            }), 403

        # 자기 자신 삭제 방지
        if user_id == request.user_id:
            return jsonify({
                'success': False,
                'error': 'Cannot delete your own account'
            }), 400

        # 사용자 삭제
        success = user_manager.delete_user(user_id)

        if not success:
            return jsonify({
                'success': False,
                'error': 'User not found'
            }), 404

        logger.info(f"[AUTH] User deleted: {user_id} by {request.user_id}")

        return jsonify({
            'success': True,
            'message': 'User deleted successfully'
        })

    except Exception as e:
        logger.error(f"[AUTH] Delete user error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    gpu_status, status_code = forward_to_gpu('/api/health')

    return jsonify({
        'status': 'ok' if status_code == 200 else 'degraded',
        'message': 'Proxy backend is running',
        'gpu_server': GPU_SERVER_URL,
        'gpu_status': gpu_status
    })


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """GPU 서버 통계 조회 (프록시)"""
    gpu_stats, status_code = forward_to_gpu('/api/stats')

    if status_code == 200:
        return jsonify(gpu_stats), 200
    else:
        # GPU 서버가 응답하지 않으면 기본값 반환
        return jsonify({
            'success': False,
            'error': 'GPU server not available'
        }), 503


@app.route('/api/projects', methods=['GET'])
@require_auth
def list_projects():
    """프로젝트 목록 조회 (사용자별)"""
    import json
    from pathlib import Path

    try:
        # 사용자 정보 조회
        user_info = user_manager.get_user_info(request.user_id)
        projects_dir = Path(BASE_PROJECTS_DIR) / request.user_id

        if not projects_dir.exists():
            return jsonify({
                'success': True,
                'projects': []
            })

        # 프로젝트 목록 읽기
        projects = []
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                project_file = project_dir / 'project.json'
                if project_file.exists():
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)

                        # 각 비디오의 어노테이션 수 계산
                        annotations_dir = project_dir / 'annotations'
                        if 'videos' in project_data and annotations_dir.exists():
                            for video in project_data['videos']:
                                video_id = video.get('video_id')
                                if video_id:
                                    video_annotations_dir = annotations_dir / video_id

                                    # 모든 사용자의 어노테이션을 합산
                                    total_annotation_count = 0
                                    all_annotated_frames = set()

                                    if video_annotations_dir.exists():
                                        try:
                                            # 디렉토리 내의 모든 JSON 파일 찾기 (사용자별 어노테이션)
                                            for json_file in video_annotations_dir.glob('*.json'):
                                                # .backup 파일은 제외
                                                if json_file.stem.endswith('.backup') or 'before_fix' in json_file.name:
                                                    continue

                                                try:
                                                    with open(json_file, 'r', encoding='utf-8') as vf:
                                                        video_data = json.load(vf)

                                                        # 어노테이션 딕셔너리 가져오기
                                                        annotations = video_data.get('annotations', {})

                                                        # 프레임 번호 추가
                                                        all_annotated_frames.update(annotations.keys())

                                                        # 어노테이션 개수 계산
                                                        for frame_annotations in annotations.values():
                                                            total_annotation_count += len(frame_annotations)
                                                except Exception as e:
                                                    logger.warning(f"[PROJECT] Failed to load annotation file {json_file}: {e}")

                                            # 어노테이션된 프레임 수 (중복 제거)
                                            video['annotated_frames'] = len(all_annotated_frames)

                                            # 총 어노테이션 수
                                            video['annotations'] = total_annotation_count

                                        except Exception as e:
                                            logger.warning(f"[PROJECT] Failed to process annotations for {video_id}: {e}")
                                            video['annotated_frames'] = 0
                                            video['annotations'] = 0
                                    else:
                                        video['annotated_frames'] = 0
                                        video['annotations'] = 0

                        projects.append(project_data)

        logger.info(f"[PROJECT] User {request.user_id} listed {len(projects)} projects")

        return jsonify({
            'success': True,
            'projects': projects
        })

    except Exception as e:
        logger.error(f"[PROJECT] List projects error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/shared', methods=['GET'])
@require_auth
def list_shared_projects():
    """다른 사용자들의 프로젝트 목록 조회 (읽기 전용)"""
    import json
    from pathlib import Path

    try:
        # 모든 사용자의 프로젝트 조회 (자신의 프로젝트 제외)
        all_projects = []
        projects_base = Path(BASE_PROJECTS_DIR)

        if not projects_base.exists():
            return jsonify({
                'success': True,
                'projects': []
            })

        # 모든 사용자 디렉토리 순회
        for user_dir in projects_base.iterdir():
            if not user_dir.is_dir():
                continue

            user_id = user_dir.name

            # 자신의 프로젝트는 제외
            if user_id == request.user_id:
                continue

            # 사용자 정보 조회
            user_info = user_manager.get_user_info(user_id)
            owner_name = user_info.get('full_name', user_id) if user_info else user_id

            # 사용자의 프로젝트 읽기
            for project_dir in user_dir.iterdir():
                if project_dir.is_dir():
                    project_file = project_dir / 'project.json'
                    if project_file.exists():
                        with open(project_file, 'r', encoding='utf-8') as f:
                            project_data = json.load(f)
                            # 소유자 정보 추가
                            project_data['owner_id'] = user_id
                            project_data['owner_name'] = owner_name
                            project_data['is_shared'] = True  # 공유 프로젝트 표시
                            all_projects.append(project_data)

        logger.info(f"[PROJECT] User {request.user_id} listed {len(all_projects)} shared projects")

        return jsonify({
            'success': True,
            'projects': all_projects
        })

    except Exception as e:
        logger.error(f"[PROJECT] List shared projects error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects', methods=['POST'])
@require_auth
def create_project():
    """프로젝트 생성 (사용자별)"""
    import json
    from datetime import datetime
    from pathlib import Path

    try:
        data = request.get_json()
        project_name = data.get('name')
        classes_str = data.get('classes', [])
        worker = data.get('worker', '')

        if not project_name:
            return jsonify({
                'success': False,
                'error': 'Project name is required'
            }), 400

        # 클래스 목록 처리 (문자열을 배열로 변환)
        if isinstance(classes_str, str):
            classes = [c.strip() for c in classes_str.split(',') if c.strip()]
        elif isinstance(classes_str, list):
            classes = classes_str
        else:
            classes = []

        # 사용자 정보 조회
        user_info = user_manager.get_user_info(request.user_id)
        projects_dir = Path(BASE_PROJECTS_DIR) / request.user_id

        # 프로젝트 ID 생성 (타임스탬프 기반)
        project_id = f"{project_name.replace(' ', '_')}_{int(datetime.now().timestamp())}"

        # 프로젝트 디렉토리 생성
        project_dir = projects_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)

        # 프로젝트 메타데이터 저장
        project_data = {
            'id': project_id,
            'name': project_name,
            'user_id': request.user_id,
            'created_at': datetime.now().isoformat(),
            'classes': classes,
            'videos': []
        }

        # 작업자 정보 추가 (선택사항)
        if worker:
            project_data['worker'] = worker

        project_file = project_dir / 'project.json'
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[PROJECT] User {request.user_id} created project: {project_id}")

        # 응답에 path 포함
        return jsonify({
            'success': True,
            'project': {
                **project_data,
                'path': str(project_dir)
            }
        })

    except Exception as e:
        logger.error(f"[PROJECT] Create project error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>', methods=['GET'])
@require_auth
def get_project(project_id):
    """프로젝트 상세 정보 (자신의 프로젝트 또는 다른 사용자의 프로젝트 읽기)"""
    import json
    from pathlib import Path

    try:
        # 1. 먼저 자신의 프로젝트에서 찾기
        user_info = user_manager.get_user_info(request.user_id)
        my_projects_dir = Path(BASE_PROJECTS_DIR) / request.user_id
        project_file = my_projects_dir / project_id / 'project.json'
        project_dir = my_projects_dir / project_id

        # 2. 자신의 프로젝트에 없으면 다른 사용자의 프로젝트에서 찾기
        if not project_file.exists():
            logger.info(f"[PROJECT] User {request.user_id} trying to access shared project {project_id}")

            # 모든 사용자 디렉토리에서 프로젝트 검색
            projects_base = Path(BASE_PROJECTS_DIR)
            found = False

            for user_dir in projects_base.iterdir():
                if not user_dir.is_dir():
                    continue

                user_id = user_dir.name
                if user_id == request.user_id:
                    continue  # 자신의 디렉토리는 이미 확인했음

                candidate_file = user_dir / project_id / 'project.json'
                if candidate_file.exists():
                    project_file = candidate_file
                    project_dir = user_dir / project_id
                    found = True
                    logger.info(f"[PROJECT] Found shared project {project_id} owned by {user_id}")
                    break

            if not found:
                return jsonify({
                    'success': False,
                    'error': 'Project not found'
                }), 404

        # 프로젝트 데이터 로드
        with open(project_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # annotations_dir 추가 (project_dir은 이미 설정됨)
        annotations_dir = project_dir / 'annotations'

        # 각 비디오의 어노테이션 수 계산 (총 프레임 수는 보존)
        if 'videos' in project_data and annotations_dir.exists():
            for video in project_data['videos']:
                video_id = video.get('video_id')
                if video_id:
                    video_annotations_dir = annotations_dir / video_id

                    # 모든 사용자의 어노테이션을 합산
                    total_annotation_count = 0
                    all_annotated_frames = set()

                    if video_annotations_dir.exists():
                        try:
                            # 디렉토리 내의 모든 JSON 파일 찾기 (사용자별 어노테이션)
                            for json_file in video_annotations_dir.glob('*.json'):
                                # .backup 파일은 제외
                                if json_file.stem.endswith('.backup') or 'before_fix' in json_file.name:
                                    continue

                                try:
                                    with open(json_file, 'r', encoding='utf-8') as vf:
                                        video_data = json.load(vf)

                                        # 어노테이션 딕셔너리 가져오기
                                        annotations = video_data.get('annotations', {})

                                        # 프레임 번호 추가
                                        all_annotated_frames.update(annotations.keys())

                                        # 어노테이션 개수 계산
                                        for frame_annotations in annotations.values():
                                            total_annotation_count += len(frame_annotations)
                                except Exception as e:
                                    logger.warning(f"[PROJECT] Failed to load annotation file {json_file}: {e}")

                            # 어노테이션된 프레임 수 (중복 제거)
                            video['annotated_frames'] = len(all_annotated_frames)

                            # 총 어노테이션 수
                            video['annotations'] = total_annotation_count

                        except Exception as e:
                            logger.warning(f"[PROJECT] Failed to process annotations for {video_id}: {e}")
                            video['annotated_frames'] = 0
                            video['annotations'] = 0
                    else:
                        video['annotated_frames'] = 0
                        video['annotations'] = 0

        # 프로젝트 통계 계산
        videos = project_data.get('videos', [])
        total_videos = len(videos)
        # status가 'completed'인 비디오만 완료로 카운팅
        annotated_videos = sum(1 for v in videos if v.get('status') == 'completed')
        total_annotations = sum(v.get('annotations', 0) for v in videos)
        annotated_frames = sum(v.get('annotated_frames', 0) for v in videos)

        stats = {
            'total_videos': total_videos,
            'annotated_videos': annotated_videos,
            'total_annotations': total_annotations,
            'annotated_frames': annotated_frames,
            'datasets': 0  # TODO: 데이터셋 기능 구현 시 업데이트
        }

        return jsonify({
            'success': True,
            'project': {
                **project_data,
                'stats': stats,
                'project_dir': str(project_dir.resolve())
            }
        })

    except Exception as e:
        logger.error(f"[PROJECT] Get project error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>', methods=['PATCH'])
@require_auth
def update_project(project_id):
    """프로젝트 업데이트 (클래스 목록 등)"""
    import json
    from pathlib import Path

    try:
        # 사용자 정보 조회
        user_info = user_manager.get_user_info(request.user_id)
        projects_dir = Path(BASE_PROJECTS_DIR) / request.user_id

        # 프로젝트 파일 경로
        project_file = projects_dir / project_id / 'project.json'

        if not project_file.exists():
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 프로젝트 데이터 로드
        with open(project_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 권한 확인 (프로젝트 소유자만 수정 가능)
        if project_data.get('user_id') != request.user_id:
            logger.warning(f"[PROJECT] Unauthorized update attempt by {request.user_id} to project {project_id}")
            return jsonify({
                'success': False,
                'error': 'Access denied'
            }), 403

        # 업데이트할 데이터 가져오기
        update_data = request.json

        # 프로젝트 이름 업데이트
        if 'name' in update_data:
            new_name = update_data['name'].strip()
            if not new_name:
                return jsonify({
                    'success': False,
                    'error': 'Project name cannot be empty'
                }), 400

            project_data['name'] = new_name
            logger.info(f"[PROJECT] Updated name for project {project_id}: {new_name}")

        # 클래스 목록 업데이트
        if 'classes' in update_data:
            classes = update_data['classes']

            # 검증
            if not isinstance(classes, list) or len(classes) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Classes must be a non-empty list'
                }), 400

            project_data['classes'] = classes
            logger.info(f"[PROJECT] Updated classes for project {project_id}: {classes}")

        # 프로젝트 파일 저장
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[PROJECT] User {request.user_id} updated project: {project_id}")

        return jsonify({
            'success': True,
            'message': 'Project updated successfully',
            'project': project_data
        })

    except Exception as e:
        logger.error(f"[PROJECT] Update project error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>', methods=['DELETE'])
@require_auth
def delete_project(project_id):
    """프로젝트 삭제 (사용자별, 소유자만 가능)"""
    import json
    import shutil
    from pathlib import Path

    try:
        # 사용자 정보 조회
        user_info = user_manager.get_user_info(request.user_id)
        projects_dir = Path(BASE_PROJECTS_DIR) / request.user_id

        # 프로젝트 디렉토리 경로
        project_dir = projects_dir / project_id
        project_file = project_dir / 'project.json'

        if not project_file.exists():
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 프로젝트 데이터 로드하여 소유자 확인
        with open(project_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 권한 확인 (프로젝트 소유자만 삭제 가능)
        if project_data.get('user_id') != request.user_id:
            logger.warning(f"[PROJECT] Unauthorized delete attempt by {request.user_id} to project {project_id}")
            return jsonify({
                'success': False,
                'error': 'Access denied'
            }), 403

        # 프로젝트 디렉토리 전체 삭제
        if project_dir.exists():
            shutil.rmtree(project_dir)

        logger.info(f"[PROJECT] User {request.user_id} deleted project: {project_id}")

        return jsonify({
            'success': True,
            'message': 'Project deleted successfully'
        })

    except Exception as e:
        logger.error(f"[PROJECT] Delete project error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ========================================
# 관리자 전용 엔드포인트
# ========================================

@app.route('/api/admin/projects', methods=['GET'])
@require_admin
def admin_list_all_projects():
    """관리자용: 모든 사용자의 프로젝트 목록 조회"""
    import json
    from pathlib import Path

    try:
        base_dir = Path(BASE_PROJECTS_DIR)
        all_projects = []

        if not base_dir.exists():
            return jsonify({
                'success': True,
                'projects': []
            })

        # 모든 사용자 디렉토리 순회
        for user_dir in base_dir.iterdir():
            if not user_dir.is_dir():
                continue

            user_id = user_dir.name
            user_info = user_manager.get_user_info(user_id)

            # 사용자의 모든 프로젝트 순회
            for project_dir in user_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                project_file = project_dir / 'project.json'
                if not project_file.exists():
                    continue

                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)

                    # 프로젝트에 소유자 정보 추가
                    project_data['owner_id'] = user_id
                    project_data['owner_name'] = user_info.get('full_name') if user_info else None

                    # 비디오 및 어노테이션 개수 계산
                    annotations_dir = project_dir / 'annotations'
                    if annotations_dir.exists():
                        # 비디오 디렉토리 개수 계산
                        video_dirs = [d for d in annotations_dir.iterdir() if d.is_dir() and d.name.startswith('video_')]
                        video_count = len(video_dirs)
                        project_data['video_count'] = video_count

                        # 어노테이션 개수 계산
                        annotation_count = 0
                        for video_dir in video_dirs:
                            annotations_file = video_dir / 'annotations.json'
                            if annotations_file.exists():
                                try:
                                    with open(annotations_file, 'r', encoding='utf-8') as vf:
                                        video_data = json.load(vf)
                                        annotations = video_data.get('annotations', {})
                                        # 모든 프레임의 어노테이션 수 합산
                                        annotation_count += sum(len(frame_annotations) for frame_annotations in annotations.values())
                                except:
                                    pass
                        project_data['annotation_count'] = annotation_count
                    else:
                        project_data['video_count'] = 0
                        project_data['annotation_count'] = 0

                    all_projects.append(project_data)

                except Exception as e:
                    logger.warning(f"[ADMIN] Failed to load project {project_dir}: {e}")
                    continue

        logger.info(f"[ADMIN] User {request.user_id} listed {len(all_projects)} projects from all users")

        return jsonify({
            'success': True,
            'projects': all_projects
        })

    except Exception as e:
        logger.error(f"[ADMIN] List all projects error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/admin/projects/<project_id>', methods=['GET'])
@require_admin
def admin_get_project(project_id):
    """관리자용: 특정 프로젝트 상세 조회 (모든 사용자 검색)"""
    import json
    from pathlib import Path

    try:
        base_dir = Path(BASE_PROJECTS_DIR)

        # 모든 사용자 디렉토리에서 프로젝트 찾기
        for user_dir in base_dir.iterdir():
            if not user_dir.is_dir():
                continue

            project_dir = user_dir / project_id
            project_file = project_dir / 'project.json'

            if project_file.exists():
                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)

                    user_id = user_dir.name
                    user_info = user_manager.get_user_info(user_id)

                    # 소유자 정보 추가
                    project_data['owner_id'] = user_id
                    project_data['owner_name'] = user_info.get('full_name') if user_info else None
                    project_data['path'] = str(project_dir.resolve())

                    # 비디오 및 어노테이션 개수 계산
                    annotations_dir = project_dir / 'annotations'
                    if annotations_dir.exists():
                        video_dirs = [d for d in annotations_dir.iterdir() if d.is_dir() and d.name.startswith('video_')]
                        video_count = len(video_dirs)
                        project_data['video_count'] = video_count

                        annotation_count = 0
                        for video_dir in video_dirs:
                            annotations_file = video_dir / 'annotations.json'
                            if annotations_file.exists():
                                try:
                                    with open(annotations_file, 'r', encoding='utf-8') as vf:
                                        video_data = json.load(vf)
                                        annotations = video_data.get('annotations', {})
                                        annotation_count += sum(len(frame_annotations) for frame_annotations in annotations.values())
                                except:
                                    pass
                        project_data['annotation_count'] = annotation_count
                    else:
                        project_data['video_count'] = 0
                        project_data['annotation_count'] = 0

                    logger.info(f"[ADMIN] User {request.user_id} viewed project {project_id} (owner: {user_id})")

                    return jsonify({
                        'success': True,
                        'project': project_data
                    })

                except Exception as e:
                    logger.error(f"[ADMIN] Failed to load project {project_id}: {e}")
                    return jsonify({
                        'success': False,
                        'error': str(e)
                    }), 500

        # 프로젝트를 찾지 못함
        return jsonify({
            'success': False,
            'error': 'Project not found'
        }), 404

    except Exception as e:
        logger.error(f"[ADMIN] Get project error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/admin/projects/<project_id>', methods=['DELETE'])
@require_admin
def admin_delete_project(project_id):
    """관리자용: 모든 사용자의 프로젝트 삭제 (관리자 권한)"""
    import json
    import shutil
    from pathlib import Path

    try:
        base_dir = Path(BASE_PROJECTS_DIR)

        # 모든 사용자 디렉토리에서 프로젝트 찾기
        project_found = False
        owner_user_id = None

        for user_dir in base_dir.iterdir():
            if not user_dir.is_dir():
                continue

            project_dir = user_dir / project_id
            project_file = project_dir / 'project.json'

            if project_file.exists():
                project_found = True
                owner_user_id = user_dir.name

                # 프로젝트 정보 로드
                with open(project_file, 'r', encoding='utf-8') as f:
                    project_data = json.load(f)

                # 프로젝트 디렉토리 삭제
                shutil.rmtree(project_dir)

                logger.info(f"[ADMIN] User {request.user_id} deleted project {project_id} (owner: {owner_user_id})")

                return jsonify({
                    'success': True,
                    'message': f'Project deleted successfully (owner: {owner_user_id})'
                })

        if not project_found:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

    except Exception as e:
        logger.error(f"[ADMIN] Delete project error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/admin/completed-videos', methods=['GET'])
@require_admin
def admin_get_completed_videos():
    """관리자용: 모든 사용자의 완료된 비디오 목록 조회 (사용자별 그룹)"""
    import json
    from pathlib import Path

    try:
        base_dir = Path(BASE_PROJECTS_DIR)

        # 사용자별로 완료된 비디오 저장
        users_completed_videos = []

        # 모든 사용자 디렉토리 탐색
        for user_dir in base_dir.iterdir():
            if not user_dir.is_dir():
                continue

            user_id = user_dir.name
            user_info = user_manager.get_user_info(user_id)
            user_name = user_info.get('full_name') if user_info else user_id

            user_completed_videos = []

            # 해당 사용자의 모든 프로젝트 탐색
            for project_dir in user_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                project_file = project_dir / 'project.json'
                if not project_file.exists():
                    continue

                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)

                    project_id = project_data.get('id', project_dir.name)
                    project_name = project_data.get('name', project_id)

                    # 모든 비디오 탐색
                    videos = project_data.get('videos', [])
                    for video in videos:
                        video_id = video.get('video_id')
                        if not video_id:
                            continue

                        # 어노테이션 파일 확인
                        annotation_file = project_dir / 'annotations' / video_id / 'annotations.json'
                        annotation_count = 0
                        frame_count_with_annotations = 0

                        if annotation_file.exists():
                            try:
                                with open(annotation_file, 'r', encoding='utf-8') as f:
                                    annotation_data = json.load(f)
                                    annotations = annotation_data.get('annotations', {})
                                    # 프레임별 어노테이션 개수 계산
                                    frame_count_with_annotations = len(annotations)
                                    for frame_annotations in annotations.values():
                                        annotation_count += len(frame_annotations)
                            except Exception as e:
                                logger.warning(f"Error reading annotations for {video_id}: {e}")

                        # 어노테이션이 있는 비디오만 포함
                        if annotation_count > 0 or video.get('complete', False):
                            user_completed_videos.append({
                                'video_id': video_id,
                                'video_name': video.get('filename', video_id),
                                'project_id': project_id,
                                'project_name': project_name,
                                'annotations': annotation_count,
                                'frame_count': frame_count_with_annotations,
                                'complete': video.get('complete', False),
                                'total_frames': video.get('total_frames', 0)
                            })

                except Exception as e:
                    logger.error(f"Error reading project {project_dir.name}: {e}")
                    continue

            # 이 사용자에게 완료된 비디오가 있으면 추가
            if user_completed_videos:
                users_completed_videos.append({
                    'user_id': user_id,
                    'user_name': user_name,
                    'completed_videos_count': len(user_completed_videos),
                    'videos': user_completed_videos
                })

        logger.info(f"[ADMIN] User {request.user_id} requested completed videos summary")

        return jsonify({
            'success': True,
            'users': users_completed_videos,
            'total_users': len(users_completed_videos),
            'total_completed_videos': sum(u['completed_videos_count'] for u in users_completed_videos)
        })

    except Exception as e:
        logger.error(f"Error getting completed videos: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/admin/annotation-stats', methods=['GET'])
@require_admin
def admin_get_annotation_stats():
    """관리자용: 전체 어노테이션 통계 조회"""
    import json
    from pathlib import Path

    try:
        base_dir = Path(BASE_PROJECTS_DIR)

        # 전체 통계
        total_stats = {
            'total_users': 0,
            'total_projects': 0,
            'total_videos': 0,
            'total_annotated_videos': 0,
            'total_annotations': 0,
            'total_annotated_frames': 0
        }

        # 사용자별 통계
        user_stats_list = []

        # 모든 사용자 디렉토리 탐색
        for user_dir in base_dir.iterdir():
            if not user_dir.is_dir():
                continue

            user_id = user_dir.name
            user_info = user_manager.get_user_info(user_id)
            user_name = user_info.get('full_name') if user_info else user_id

            user_stats = {
                'user_id': user_id,
                'user_name': user_name,
                'projects': 0,
                'videos': 0,
                'annotated_videos': 0,
                'annotations': 0,
                'annotated_frames': 0
            }

            # 해당 사용자의 모든 프로젝트 탐색
            for project_dir in user_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                project_file = project_dir / 'project.json'
                if not project_file.exists():
                    continue

                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)

                    user_stats['projects'] += 1

                    videos = project_data.get('videos', [])
                    user_stats['videos'] += len(videos)

                    for video in videos:
                        annotations_count = video.get('annotations', 0)
                        frame_count = video.get('frame_count', 0)

                        if annotations_count > 0:
                            user_stats['annotated_videos'] += 1
                            user_stats['annotations'] += annotations_count
                            user_stats['annotated_frames'] += frame_count

                except Exception as e:
                    logger.error(f"Error reading project {project_dir.name}: {e}")
                    continue

            # 통계가 있는 사용자만 추가
            if user_stats['projects'] > 0:
                user_stats_list.append(user_stats)
                total_stats['total_users'] += 1
                total_stats['total_projects'] += user_stats['projects']
                total_stats['total_videos'] += user_stats['videos']
                total_stats['total_annotated_videos'] += user_stats['annotated_videos']
                total_stats['total_annotations'] += user_stats['annotations']
                total_stats['total_annotated_frames'] += user_stats['annotated_frames']

        logger.info(f"[ADMIN] User {request.user_id} requested annotation stats")

        return jsonify({
            'success': True,
            'total_stats': total_stats,
            'user_stats': user_stats_list
        })

    except Exception as e:
        logger.error(f"Error getting annotation stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos', methods=['POST'])
@require_auth
def add_video(project_id):
    """비디오 추가 (JSON 또는 FormData, NAS 비디오 참조 지원)"""
    import json
    from pathlib import Path
    import re

    def safe_filename(filename):
        """한글을 포함한 파일명을 안전하게 처리"""
        # 경로 구분자 제거
        filename = filename.replace('/', '_').replace('\\', '_')
        # 위험한 문자만 제거 (. _ - 한글 영문 숫자는 유지)
        filename = re.sub(r'[^\w\s.-]', '', filename, flags=re.UNICODE)
        # 연속된 공백을 하나로
        filename = re.sub(r'\s+', ' ', filename)
        # 앞뒤 공백 제거
        filename = filename.strip()
        # 빈 파일명 방지
        if not filename:
            filename = 'video'
        return filename

    try:
        # 프로젝트 디렉토리 찾기 (관리자는 모든 사용자 폴더 검색)
        project_dir = find_project_dir(project_id, request.user_id)
        if not project_dir:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        project_file = project_dir / 'project.json'

        if not project_file.exists():
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 프로젝트 데이터 로드하여 소유자 확인 (관리자는 모든 프로젝트 접근 가능)
        with open(project_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 소유자가 아니면서 관리자도 아닌 경우만 거부
        user_info = user_manager.get_user_info(request.user_id)
        is_admin = user_info and user_info.get('role') == 'admin'

        if project_data.get('user_id') != request.user_id and not is_admin:
            logger.warning(f"[VIDEO] Unauthorized access attempt by {request.user_id} to project {project_id}")
            return jsonify({
                'success': False,
                'error': 'Access denied'
            }), 403

        # Content-Type 확인
        content_type = request.content_type or ''

        if 'multipart/form-data' in content_type:
            # FormData 업로드 (웹 브라우저)
            if 'video' not in request.files:
                return jsonify({
                    'success': False,
                    'error': 'No video file provided'
                }), 400

            video_file = request.files['video']
            if video_file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'Empty filename'
                }), 400

            # 파일명 안전하게 처리 (한글 지원)
            filename = safe_filename(video_file.filename)

            # 부모 디렉토리명 가져오기 (선택사항)
            parent_dir = request.form.get('parent_dir', '').strip()

            # 고유한 video_id 생성
            import time
            video_id = f"video_{int(time.time() * 1000)}"

            # 비디오별 고유 디렉토리 생성
            video_dir = project_dir / 'videos' / video_id
            video_dir.mkdir(parents=True, exist_ok=True)

            # 비디오 파일 저장
            video_path = video_dir / filename
            video_file.save(str(video_path))

            logger.info(f"[VIDEO] User {request.user_id} uploaded video to project {project_id}: {filename} (video_id: {video_id}, parent_dir: {parent_dir or 'N/A'})")

            # GPU 서버로 비디오 경로 전달 (JSON)
            gpu_data = {
                'video_path': str(video_path),
                'project_dir': str(project_dir),
                'video_id': video_id  # backend proxy에서 생성한 video_id 전달
            }
            if parent_dir:
                gpu_data['parent_dir'] = parent_dir

            data, status_code = forward_to_gpu(
                f'/api/projects/{project_id}/videos',
                method='POST',
                json=gpu_data
            )
            return jsonify(data), status_code
        else:
            # JSON 요청 (Electron 또는 NAS 비디오 참조)
            request_data = request.json

            # NAS 비디오 참조 처리
            nas_video_path = request_data.get('nas_video_path')
            if nas_video_path:
                # 보안: 허용된 NAS 경로인지 확인
                NAS_ALLOWED_PATHS = [
                    '/home/intu/nas2_kwater/Videos/SAHARA',
                    '/home/intu/nas2_kwater/Videos/관내시경영상'
                ]

                is_allowed = any(nas_video_path.startswith(allowed_path)
                               for allowed_path in NAS_ALLOWED_PATHS)

                if not is_allowed:
                    logger.warning(f"[VIDEO] Unauthorized NAS path access attempt: {nas_video_path}")
                    return jsonify({
                        'success': False,
                        'error': 'Access to this NAS path is not allowed'
                    }), 403

                if not os.path.exists(nas_video_path):
                    return jsonify({
                        'success': False,
                        'error': 'NAS video file not found'
                    }), 404

                # NAS 비디오 메타데이터 추출
                nas_metadata = request_data.get('nas_metadata', {})

                logger.info(f"[VIDEO] User {request.user_id} added NAS video to project {project_id}: {nas_video_path}")

                # GPU 서버로 NAS 비디오 경로 전달 (복사하지 않음)
                gpu_data = {
                    'nas_video_path': nas_video_path,
                    'nas_metadata': nas_metadata,
                    'project_dir': str(project_dir),
                    'is_nas_reference': True  # NAS 참조 플래그
                }

                data, status_code = forward_to_gpu(
                    f'/api/projects/{project_id}/videos',
                    method='POST',
                    json=gpu_data
                )
                return jsonify(data), status_code
            else:
                # 기존 JSON 요청 (Electron 업로드)
                request_data['project_dir'] = str(project_dir)
                data, status_code = forward_to_gpu(
                    f'/api/projects/{project_id}/videos',
                    method='POST',
                    json=request_data
                )
                return jsonify(data), status_code

    except Exception as e:
        logger.error(f"[VIDEO] Add video error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>', methods=['GET'])
@require_auth
def get_video(project_id, video_id):
    """비디오 상세 정보"""
    from pathlib import Path
    import json

    # 프로젝트 디렉토리 찾기 (관리자는 모든 사용자 폴더 검색)
    project_dir = find_project_dir(project_id, request.user_id)
    if not project_dir:
        return jsonify({'success': False, 'error': 'Project not found'}), 404

    # project.json에서 비디오 정보 읽기
    project_file = project_dir / 'project.json'

    if not project_file.exists():
        return jsonify({'success': False, 'error': 'Project not found'}), 404

    try:
        with open(project_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 비디오 찾기
        video = None
        for v in project_data.get('videos', []):
            if v.get('video_id') == video_id:
                video = v
                break

        if not video:
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        # 비디오 정보 반환 (nas_metadata 포함)
        return jsonify({
            'success': True,
            'video': {
                'id': video.get('video_id'),
                'filename': video.get('filename'),
                'path': video.get('video_path'),
                'total_frames': video.get('total_frames'),
                'frame_count': video.get('total_frames'),  # 호환성
                'status': video.get('status', 'in_progress'),  # 비디오 상태 추가
                'nas_metadata': video.get('nas_metadata', {}),
                'is_nas_reference': video.get('is_nas_reference', False)
            }
        }), 200

    except Exception as e:
        logger.error(f"[VIDEO] Error getting video info: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/projects/<project_id>/videos/<video_id>/stream', methods=['GET'])
def stream_video(project_id, video_id):
    """비디오 파일 스트리밍 (GPU 서버에서 비디오 정보를 가져와 직접 스트리밍)"""
    import os
    from pathlib import Path

    try:
        # 세션 ID 확인 (쿼리 파라미터, 헤더, 또는 쿠키에서)
        session_id = request.args.get('session_id') or request.headers.get('X-Session-ID') or request.cookies.get('session_id')

        if not session_id:
            logger.warning("[STREAM] No session ID provided")
            return jsonify({'success': False, 'error': 'Authentication required'}), 401

        # 세션 검증
        user_id = user_manager.validate_session(session_id)
        if not user_id:
            logger.warning(f"[STREAM] Invalid or expired session")
            return jsonify({'success': False, 'error': 'Invalid or expired session'}), 401

        # 프로젝트 디렉토리 찾기 (관리자는 모든 사용자 폴더 검색)
        project_dir = find_project_dir(project_id, user_id)
        if not project_dir:
            logger.warning(f"[STREAM] Project not found: {project_id}")
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        # GPU 서버에서 비디오 정보 가져오기
        video_info_url = f"{GPU_SERVER_URL}/api/projects/{project_id}/videos/{video_id}"
        logger.info(f"Getting video info from: {video_info_url}")

        response = requests.get(video_info_url, params={'project_dir': str(project_dir)}, timeout=10)

        if response.status_code != 200:
            logger.error(f"Failed to get video info: {response.status_code}")
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        video_data = response.json()
        if not video_data.get('success') or not video_data.get('video'):
            logger.error(f"Invalid video data response")
            return jsonify({'success': False, 'error': 'Invalid video data'}), 500

        # 비디오 파일 경로 가져오기
        video_path = video_data['video'].get('path')
        if not video_path:
            logger.error("No video path in response")
            return jsonify({'success': False, 'error': 'Video path not found'}), 404

        video_path = Path(video_path)

        # 웹 호환 버전이 있는지 확인 (Videos -> Videos_web, 확장자 .mp4)
        web_video_path = None
        if '/Videos/' in str(video_path):
            # Videos를 Videos_web으로 변경하고 확장자를 .mp4로
            web_path_str = str(video_path).replace('/Videos/', '/Videos_web/')
            web_video_path = Path(web_path_str).with_suffix('.mp4')

            if web_video_path.exists():
                logger.info(f"Using web-compatible version: {web_video_path}")
                video_path = web_video_path
            else:
                logger.info(f"Web version not found, using original: {video_path}")

        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return jsonify({'success': False, 'error': 'Video file not found'}), 404

        logger.info(f"Streaming video from: {video_path}")

        # 비디오 파일 직접 스트리밍
        from flask import send_file
        return send_file(
            str(video_path),
            mimetype='video/mp4',
            as_attachment=False,
            conditional=True  # Range requests 지원
        )

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Cannot connect to GPU server: {e}")
        return jsonify({
            'success': False,
            'error': 'Cannot connect to GPU server'
        }), 503
    except Exception as e:
        logger.error(f"Error streaming video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>/frame/<int:frame_number>', methods=['GET'])
@require_auth
def get_video_frame(project_id, video_id, frame_number):
    """비디오 프레임 이미지 가져오기"""
    logger.info(f"[FRAME] get_video_frame called: project={project_id}, video={video_id}, frame={frame_number}")
    from pathlib import Path

    # 프로젝트 디렉토리 찾기 (관리자는 모든 사용자 폴더 검색)
    project_dir = find_project_dir(project_id, request.user_id)
    if not project_dir:
        return jsonify({'success': False, 'error': 'Project not found'}), 404

    url = f"{GPU_SERVER_URL}/api/projects/{project_id}/videos/{video_id}/frame/{frame_number}"
    logger.info(f"Forwarding frame request to {url}")

    try:
        response = requests.get(url, params={'project_dir': str(project_dir)}, timeout=90)

        if response.status_code == 200:
            return Response(
                response.content,
                content_type='image/jpeg',
                headers={'Cache-Control': 'public, max-age=3600'}
            )
        else:
            logger.error(f"GPU server returned status code: {response.status_code}")
            return jsonify({'success': False, 'error': 'Frame not found'}), response.status_code

    except Exception as e:
        logger.error(f"Error getting frame: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>', methods=['DELETE'])
@require_auth
def remove_video(project_id, video_id):
    """비디오 제거"""
    from pathlib import Path

    # 프로젝트 디렉토리 찾기 (관리자는 모든 사용자 폴더 검색)
    project_dir = find_project_dir(project_id, request.user_id)
    if not project_dir:
        return jsonify({'success': False, 'error': 'Project not found'}), 404

    data, status_code = forward_to_gpu(
        f'/api/projects/{project_id}/videos/{video_id}',
        method='DELETE',
        params={'project_dir': str(project_dir)}
    )
    return jsonify(data), status_code


@app.route('/api/projects/<project_id>/videos/<video_id>/status', methods=['PUT'])
@require_auth
def update_video_status(project_id, video_id):
    """비디오 상태 업데이트 (in_progress, completed)"""
    import json
    from pathlib import Path

    try:
        # 프로젝트 디렉토리 찾기 (관리자는 모든 사용자 폴더 검색)
        project_dir = find_project_dir(project_id, request.user_id)
        if not project_dir:
            logger.error(f"[VIDEO STATUS] Project not found: {project_id}")
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 프로젝트 파일
        project_file = project_dir / 'project.json'

        if not project_file.exists():
            logger.error(f"[VIDEO STATUS] Project file not found: {project_file}")
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 요청 데이터
        data = request.get_json()
        new_status = data.get('status', 'in_progress')

        # 유효성 검사
        if new_status not in ['in_progress', 'completed']:
            return jsonify({
                'success': False,
                'error': 'Invalid status. Must be "in_progress" or "completed"'
            }), 400

        # 프로젝트 데이터 읽기
        with open(project_file, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 비디오 찾기 및 상태 업데이트
        video_found = False
        for video in project_data.get('videos', []):
            if video.get('video_id') == video_id:
                video['status'] = new_status
                video_found = True
                logger.info(f"[VIDEO STATUS] Updated {video_id} status to {new_status}")
                break

        if not video_found:
            logger.warning(f"[VIDEO STATUS] Video {video_id} not found in project {project_id}")
            return jsonify({
                'success': False,
                'error': 'Video not found in project'
            }), 404

        # 프로젝트 데이터 저장
        with open(project_file, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, ensure_ascii=False, indent=2)

        return jsonify({
            'success': True,
            'video_id': video_id,
            'status': new_status
        })

    except Exception as e:
        logger.error(f"[VIDEO STATUS] Error updating status: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/scan-videos', methods=['POST'])
@require_auth
def scan_videos():
    """디렉토리를 스캔하여 비디오 파일 목록 반환"""
    from pathlib import Path

    try:
        data = request.get_json()
        directory = data.get('directory')

        if not directory:
            return jsonify({
                'success': False,
                'error': 'Directory path is required'
            }), 400

        dir_path = Path(directory)

        # 디렉토리 존재 확인
        if not dir_path.exists():
            return jsonify({
                'success': False,
                'error': 'Directory does not exist'
            }), 404

        if not dir_path.is_dir():
            return jsonify({
                'success': False,
                'error': 'Path is not a directory'
            }), 400

        # 비디오 파일 확장자
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV'}

        # 디렉토리 스캔
        videos = []
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix in video_extensions:
                videos.append({
                    'name': file_path.name,
                    'path': str(file_path.resolve())
                })

        # 파일명으로 정렬
        videos.sort(key=lambda x: x['name'])

        logger.info(f"[SCAN] User {request.user_id} scanned directory: {directory} - Found {len(videos)} videos")

        return jsonify({
            'success': True,
            'videos': videos,
            'count': len(videos)
        })

    except PermissionError:
        logger.error(f"[SCAN] Permission denied accessing directory: {directory}")
        return jsonify({
            'success': False,
            'error': 'Permission denied accessing directory'
        }), 403
    except Exception as e:
        logger.error(f"[SCAN] Error scanning directory: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/nas-videos/list', methods=['GET'])
@require_auth
def list_nas_videos():
    """NAS 폴더의 비디오 목록 조회 (캐시 사용, 페이지네이션 및 필터링 지원)"""
    try:
        from video_cache_manager import VideoCacheManager

        # 캐시 매니저 초기화
        cache_manager = VideoCacheManager(cache_dir='.video_cache')

        # 쿼리 파라미터
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))  # 기본 50개씩
        nas_folder = request.args.get('folder', None)  # 폴더 필터

        # 메타데이터 필터 파라미터
        region = request.args.get('region', None)  # 지역 필터 (예: '지방', '광역')
        pipe_size = request.args.get('pipe_size', None)  # 파이프 크기 필터 (예: '300MM', '500MM')
        method = request.args.get('method', None)  # 방법 필터 (예: 'SP', 'DCIP', 'HI3P')

        # 캐시에서 모든 비디오 조회 (웹 호환 필터링을 위해 전체 조회)
        all_videos = cache_manager.get_all_cached_videos(
            nas_folder=nas_folder,
            region=region,
            pipe_size=pipe_size,
            method=method
        )

        # 웹 호환 비디오만 필터링 (변환 완료되었거나 원본이 이미 mp4인 경우)
        web_compatible_videos = []
        for video in all_videos:
            video_path = video['path']

            # 1. 원본이 이미 mp4인 경우
            if video_path.lower().endswith('.mp4'):
                video['has_web_version'] = True
                video['web_path'] = video_path
                web_compatible_videos.append(video)
                continue

            # 2. Videos_web에 변환된 버전이 있는지 확인
            if '/Videos/' in video_path:
                web_path = video_path.replace('/Videos/', '/Videos_web/').rsplit('.', 1)[0] + '.mp4'
                if os.path.exists(web_path):
                    video['has_web_version'] = True
                    video['web_path'] = web_path
                    web_compatible_videos.append(video)

        # 필터링된 전체 개수
        total_count = len(web_compatible_videos)

        # 페이지네이션 적용
        offset = (page - 1) * page_size
        videos = web_compatible_videos[offset:offset + page_size]

        # 전체 페이지 수
        total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

        # 캐시 통계
        stats = cache_manager.get_cache_stats()

        filter_info = []
        if nas_folder:
            filter_info.append(f"folder: {nas_folder}")
        if region:
            filter_info.append(f"region: {region}")
        if pipe_size:
            filter_info.append(f"pipe_size: {pipe_size}")
        if method:
            filter_info.append(f"method: {method}")
        filter_info.append(f"web-compatible only")

        logger.info(f"[NAS] User {request.user_id} requested page {page}/{total_pages}, {total_count} videos ({', '.join(filter_info)})")
        logger.info(f"[NAS] Filtered {len(all_videos)} videos -> {total_count} web-compatible videos")

        return jsonify({
            'success': True,
            'videos': videos,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            'filters': {
                'region': region,
                'pipe_size': pipe_size,
                'method': method,
                'folder': nas_folder
            },
            'cache_stats': stats
        })

    except Exception as e:
        logger.error(f"[NAS] Error listing videos: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 필터 옵션 캐시 (메모리)
_filter_options_cache = None
_filter_options_cache_time = 0

@app.route('/api/nas-videos/filter-options', methods=['GET'])
@require_auth
def get_nas_filter_options():
    """NAS 비디오 필터링에 사용 가능한 옵션 조회 (캐시 사용)"""
    global _filter_options_cache, _filter_options_cache_time

    try:
        import time

        # 캐시가 있고 10분 이내면 재사용
        if _filter_options_cache and (time.time() - _filter_options_cache_time) < 600:
            logger.info(f"[NAS] Returning cached filter options")
            return jsonify(_filter_options_cache)

        from video_cache_manager import VideoCacheManager
        import json

        logger.info(f"[NAS] Computing filter options from database...")
        cache_manager = VideoCacheManager(cache_dir='.video_cache')

        # 데이터베이스 연결
        import sqlite3
        conn = sqlite3.connect(str(cache_manager.db_path))
        cursor = conn.cursor()

        # 모든 메타데이터 조회
        cursor.execute('SELECT dir_metadata_raw, dir_metadata_parts FROM video_metadata')
        rows = cursor.fetchall()
        conn.close()

        regions = set()
        pipe_sizes = set()
        methods = set()

        for raw, parts_json in rows:
            # 빈 문자열 제거 (파싱 로직과 동일)
            parts = [p.strip() for p in raw.split('-') if p.strip()]

            # 최소 4개 파트이고 첫 번째가 숫자인 경우만 (정상 패턴)
            if len(parts) >= 4:
                try:
                    # 첫 번째 파트가 숫자인지 확인
                    int(parts[0])
                    regions.add(parts[1])
                    pipe_sizes.add(parts[2])
                    methods.add(parts[3])
                except (ValueError, IndexError):
                    pass

        result = {
            'success': True,
            'options': {
                'regions': sorted(list(regions)),
                'pipe_sizes': sorted(list(pipe_sizes)),
                'methods': sorted(list(methods))
            }
        }

        # 캐시 저장
        _filter_options_cache = result
        _filter_options_cache_time = time.time()
        logger.info(f"[NAS] Filter options cached")

        return jsonify(result)

    except Exception as e:
        logger.error(f"[NAS] Error getting filter options: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/nas-videos/thumbnail', methods=['GET'])
def get_nas_video_thumbnail():
    """NAS 비디오의 썸네일 반환 (캐시 사용)"""
    try:
        from video_cache_manager import VideoCacheManager

        video_path = request.args.get('path')

        if not video_path:
            return jsonify({'error': 'path parameter required'}), 400

        # 보안: 허용된 NAS 경로인지 확인
        if not (video_path.startswith('/home/intu/nas2_kwater/Videos/SAHARA') or
                video_path.startswith('/home/intu/nas2_kwater/Videos/관내시경영상')):
            logger.warning(f"[NAS] Unauthorized thumbnail request: {video_path}")
            return jsonify({'error': 'Forbidden'}), 403

        # 캐시 매니저 초기화
        cache_manager = VideoCacheManager(cache_dir='.video_cache')

        # 캐시된 메타데이터 확인
        cached_data = cache_manager.get_cached_metadata(video_path)

        if cached_data and cached_data.get('thumbnail_path'):
            # 절대 경로로 변환
            thumbnail_abs_path = os.path.abspath(cached_data['thumbnail_path'])

            if os.path.exists(thumbnail_abs_path):
                # 캐시된 썸네일 반환
                from flask import send_file
                return send_file(
                    thumbnail_abs_path,
                    mimetype='image/jpeg',
                    as_attachment=False,
                    max_age=3600
                )
        else:
            # 캐시 없음 - 기본 플레이스홀더 반환
            return Response(
                '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180"><rect fill="#333" width="320" height="180"/><text x="50%" y="50%" text-anchor="middle" fill="#888" font-size="14">썸네일 없음</text></svg>',
                mimetype='image/svg+xml'
            )

    except Exception as e:
        logger.error(f"[NAS] Thumbnail error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/initialize', methods=['POST'])
def initialize_ai():
    """AI 모델 초기화"""
    data, status_code = forward_to_gpu('/api/ai/initialize', method='POST', json=request.json)
    return jsonify(data), status_code


@app.route('/api/ai/inference', methods=['POST'])
def run_inference():
    """AI 추론 실행"""
    data, status_code = forward_to_gpu('/api/ai/inference', method='POST', json=request.json)
    return jsonify(data), status_code


@app.route('/api/ai/inference_box', methods=['POST'])
def run_inference_box():
    """박스 영역 AI 추론 실행"""
    data, status_code = forward_to_gpu('/api/ai/inference_box', method='POST', json=request.json)
    return jsonify(data), status_code


@app.route('/api/export/dataset', methods=['POST'])
def export_dataset():
    """학습 데이터셋 export"""
    data, status_code = forward_to_gpu('/api/export/dataset', method='POST', json=request.json)
    return jsonify(data), status_code


@app.route('/api/dataset/build', methods=['POST'])
@require_auth
def build_dataset():
    """다중 프로젝트 데이터셋 빌드"""
    import json
    from pathlib import Path

    try:
        data = request.get_json()
        videos = data.get('videos', [])
        output_dir = data.get('output_dir', 'pipe_dataset')
        split_ratio = data.get('split_ratio', '0.7,0.15,0.15')
        augment_multiplier = data.get('augment_multiplier', 0)
        format_type = data.get('format', 'yolo')

        if not videos:
            return jsonify({
                'success': False,
                'error': 'No videos selected'
            }), 400

        if format_type != 'yolo':
            return jsonify({
                'success': False,
                'error': 'Only YOLO format is supported'
            }), 400

        logger.info(f"[DATASET BUILD] Building dataset from {len(videos)} videos")
        logger.info(f"[DATASET BUILD] Output: {output_dir}, Split: {split_ratio}, Augment: {augment_multiplier}x")

        # 각 비디오의 어노테이션 수집
        annotations_data = []

        for video in videos:
            user_id = video['user_id']
            project_id = video['project_id']
            video_id = video['video_id']

            # 프로젝트 디렉토리 찾기
            project_dir = Path(BASE_PROJECTS_DIR) / user_id / project_id
            if not project_dir.exists():
                logger.warning(f"[DATASET BUILD] Project dir not found: {project_dir}")
                continue

            # 어노테이션 디렉토리
            annotations_dir = project_dir / 'annotations' / video_id
            if not annotations_dir.exists():
                logger.warning(f"[DATASET BUILD] Annotations dir not found: {annotations_dir}")
                continue

            # 모든 사용자의 어노테이션 JSON 파일 읽기
            for json_file in annotations_dir.glob('*.json'):
                if json_file.stem.endswith('.backup') or 'before_fix' in json_file.name:
                    continue

                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        anno_data = json.load(f)

                    annotations_data.append({
                        'user_id': user_id,
                        'project_id': project_id,
                        'video_id': video_id,
                        'annotations': anno_data.get('annotations', {}),
                        'video_name': video.get('video_name', video_id),
                        'project_dir': str(project_dir)
                    })

                except Exception as e:
                    logger.error(f"[DATASET BUILD] Error reading {json_file}: {e}")

        if not annotations_data:
            return jsonify({
                'success': False,
                'error': 'No annotations found in selected videos'
            }), 400

        logger.info(f"[DATASET BUILD] Collected {len(annotations_data)} annotation files")

        # GPU 서버로 전달하여 실제 빌드 수행
        build_request = {
            'annotations_data': annotations_data,
            'output_dir': output_dir,
            'split_ratio': split_ratio,
            'augment_multiplier': augment_multiplier,
            'format': format_type,
            'base_projects_dir': str(BASE_PROJECTS_DIR)
        }

        # GPU 서버에 빌드 요청
        gpu_response, status_code = forward_to_gpu('/api/dataset/build_yolo', method='POST', json=build_request)

        if status_code == 200 and gpu_response.get('success'):
            logger.info(f"[DATASET BUILD] Success: {gpu_response.get('output_dir')}")
        else:
            logger.error(f"[DATASET BUILD] Failed: {gpu_response.get('error')}")

        return jsonify(gpu_response), status_code

    except Exception as e:
        logger.error(f"[DATASET BUILD] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/polygon/generate_mask', methods=['POST'])
def generate_mask_from_polygon():
    """폴리곤에서 마스크 생성"""
    data, status_code = forward_to_gpu('/api/polygon/generate_mask', method='POST', json=request.json)
    return jsonify(data), status_code


@app.route('/api/projects/<project_id>/videos/<video_id>/annotations', methods=['POST'])
@require_auth
def save_annotations(project_id: str, video_id: str):
    """어노테이션 저장 (사용자별 파일로 분리 저장)"""
    import json
    from datetime import datetime
    from pathlib import Path

    try:
        data = request.get_json()
        annotations = data.get('annotations', {})

        # 프로젝트 디렉토리 찾기
        project_dir = find_project_dir(project_id, request.user_id)
        if not project_dir:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 어노테이션 디렉토리 생성
        annotation_dir = project_dir / 'annotations' / video_id
        annotation_dir.mkdir(parents=True, exist_ok=True)

        # 사용자 정보 조회
        user_info = user_manager.get_user_info(request.user_id)
        user_name = user_info.get('full_name', request.user_id) if user_info else request.user_id

        # 어노테이션을 소유자별로 분류하여 저장
        current_time = datetime.now().isoformat()
        annotations_by_owner = {}  # {owner_id: {frame_key: [annotations]}}

        for frame_key, frame_annotations in annotations.items():
            if isinstance(frame_annotations, list):
                for ann in frame_annotations:
                    if isinstance(ann, dict):
                        created_by = ann.get('created_by')

                        # 새로운 어노테이션인 경우 (created_by가 없음)
                        if created_by is None:
                            created_by = request.user_id
                            ann['created_by'] = request.user_id
                            ann['created_by_name'] = user_name
                            ann['created_at'] = current_time

                        # 댓글 삭제 처리: 빈 문자열이면 comment 필드 제거
                        if 'comment' in ann:
                            comment_text = ann.get('comment', '').strip()
                            if not comment_text:
                                del ann['comment']

                        # 수정 정보 업데이트 (댓글 추가/수정/삭제 포함)
                        ann['modified_by'] = request.user_id
                        ann['modified_by_name'] = user_name
                        ann['modified_at'] = current_time

                        # 소유자별로 분류
                        if created_by not in annotations_by_owner:
                            annotations_by_owner[created_by] = {}
                        if frame_key not in annotations_by_owner[created_by]:
                            annotations_by_owner[created_by][frame_key] = []
                        annotations_by_owner[created_by][frame_key].append(ann)

        # 각 소유자의 파일을 읽고 업데이트
        for owner_id, owner_annotations in annotations_by_owner.items():
            annotation_file = annotation_dir / f'{owner_id}.json'

            # 기존 파일이 있으면 읽어서 병합
            if annotation_file.exists():
                try:
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    existing_annotations = existing_data.get('annotations', {})
                except Exception as e:
                    logger.error(f"[ANNOTATION] Error reading existing file {annotation_file}: {e}")
                    existing_annotations = {}
            else:
                existing_annotations = {}

            # 업데이트할 프레임만 병합 (다른 프레임은 유지)
            for frame_key, frame_anns in owner_annotations.items():
                existing_annotations[frame_key] = frame_anns

            # 소유자 정보 조회
            owner_info = user_manager.get_user_info(owner_id)
            owner_name = owner_info.get('full_name', owner_id) if owner_info else owner_id

            # 파일 저장
            annotation_data = {
                'project_id': project_id,
                'video_id': video_id,
                'user_id': owner_id,
                'user_name': owner_name,
                'annotations': existing_annotations,
                'updated_at': current_time
            }

            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[ANNOTATION] Updated {owner_id}'s file by {request.user_id} ({user_name}) for {project_id}/{video_id}")

        # 빈 프레임 처리: 현재 사용자가 자신의 어노테이션을 모두 삭제한 경우
        if request.user_id not in annotations_by_owner:
            annotation_file = annotation_dir / f'{request.user_id}.json'
            if annotation_file.exists():
                try:
                    with open(annotation_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    # 빈 프레임도 유지 (명시적 삭제 기록)
                    annotation_data = {
                        'project_id': project_id,
                        'video_id': video_id,
                        'user_id': request.user_id,
                        'user_name': user_name,
                        'annotations': existing_data.get('annotations', {}),
                        'updated_at': current_time
                    }
                    with open(annotation_file, 'w', encoding='utf-8') as f:
                        json.dump(annotation_data, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"[ANNOTATION] Error updating empty frame: {e}")

        return jsonify({
            'success': True,
            'message': 'Annotations saved successfully',
            'file': str(annotation_file)
        })
    except Exception as e:
        logger.error(f"[ANNOTATION] Error saving annotations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>/annotations', methods=['GET'])
@require_auth
def load_annotations(project_id: str, video_id: str):
    """어노테이션 로드 (모든 사용자의 어노테이션 병합)"""
    import json
    from pathlib import Path

    try:
        # 프로젝트 디렉토리 찾기
        project_dir = find_project_dir(project_id, request.user_id)
        if not project_dir:
            return jsonify({
                'success': True,
                'annotations': {},
                'contributors': [],
                'message': 'Project not found'
            })

        # 어노테이션 디렉토리
        annotation_dir = project_dir / 'annotations' / video_id

        if not annotation_dir.exists():
            return jsonify({
                'success': True,
                'annotations': {},
                'contributors': [],
                'message': 'No annotations found'
            })

        # 모든 사용자의 어노테이션 파일 병합
        merged_annotations = {}
        contributors = []
        latest_update = None

        # 디렉토리 내의 모든 JSON 파일 읽기
        for annotation_file in annotation_dir.glob('*.json'):
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    annotation_data = json.load(f)

                # 파일 소유자 (파일명에서 추출)
                file_owner = annotation_file.stem

                user_id = annotation_data.get('user_id')
                user_name = annotation_data.get('user_name', user_id)
                updated_at = annotation_data.get('updated_at')
                user_annotations = annotation_data.get('annotations', {})

                # 이 사용자의 유효한 어노테이션 개수 추적
                user_annotation_count = 0

                # 어노테이션 병합 (레거시 데이터 필터링 포함)
                for frame_key, frame_annotations in user_annotations.items():
                    if frame_key not in merged_annotations:
                        merged_annotations[frame_key] = []
                    if isinstance(frame_annotations, list):
                        # 각 어노테이션을 필터링하여 추가
                        for ann in frame_annotations:
                            if isinstance(ann, dict):
                                created_by = ann.get('created_by')
                                # 필터링: created_by가 없거나 파일 소유자와 일치하는 경우만 포함
                                # (레거시 데이터 오염 방지: kwater2.json에 created_by="kwater1"인 경우 제외)
                                if created_by is None or created_by == file_owner:
                                    # 어노테이션 복사본 생성하여 created_by_name 추가
                                    ann_with_name = ann.copy()
                                    ann_with_name['created_by_name'] = user_name
                                    merged_annotations[frame_key].append(ann_with_name)
                                    user_annotation_count += 1
                                else:
                                    logger.debug(f"[ANNOTATION] Filtered out mismatched annotation: created_by={created_by}, file_owner={file_owner}")
                            else:
                                # dict가 아닌 경우 그대로 추가 (호환성)
                                merged_annotations[frame_key].append(ann)
                                user_annotation_count += 1

                # 기여자 목록에 추가 (실제 어노테이션이 있는 경우만)
                if user_id and user_annotation_count > 0:
                    contributors.append({
                        'user_id': user_id,
                        'user_name': user_name,
                        'updated_at': updated_at
                    })

                # 최신 업데이트 시간 추적
                if updated_at and (not latest_update or updated_at > latest_update):
                    latest_update = updated_at

            except Exception as e:
                logger.error(f"[ANNOTATION] Error loading file {annotation_file}: {e}")
                continue

        logger.info(f"[ANNOTATION] User {request.user_id} loaded merged annotations for {project_id}/{video_id} from {len(contributors)} contributor(s)")

        return jsonify({
            'success': True,
            'annotations': merged_annotations,
            'contributors': contributors,
            'updated_at': latest_update
        })
    except Exception as e:
        logger.error(f"[ANNOTATION] Error loading annotations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/comments/all', methods=['GET'])
@require_auth
def get_all_commented_annotations():
    """시스템 내 모든 코멘트가 달린 어노테이션 조회"""
    try:
        user_id = request.user_id
        commented_annotations = []

        # 모든 사용자의 프로젝트 스캔
        for user_dir in BASE_PROJECTS_DIR.iterdir():
            if not user_dir.is_dir():
                continue

            user_owner = user_dir.name

            # 각 프로젝트 스캔
            for project_dir in user_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                project_name = project_dir.name
                # project_id 추출 (마지막 underscore 뒤의 숫자)
                project_id = project_name.split('_')[-1] if '_' in project_name else project_name

                annotations_dir = project_dir / 'annotations'
                if not annotations_dir.exists():
                    continue

                # ⚡ 최적화: 프로젝트당 한 번만 discussions.json 로드
                discussions_data = load_discussions(user_owner, project_name)

                # 각 비디오 스캔
                for video_dir in annotations_dir.iterdir():
                    if not video_dir.is_dir():
                        continue

                    video_id = video_dir.name

                    # 각 어노테이션 파일 스캔
                    for json_file in video_dir.glob('*.json'):
                        if json_file.name.endswith('.backup') or json_file.name.endswith('_fix') or json_file.name.endswith('.before_fix'):
                            continue

                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                        except Exception as e:
                            logger.warning(f"Error reading {json_file}: {e}")
                            continue

                        file_owner = json_file.stem

                        # JSON 파일의 user_id를 가져옴 (file_owner보다 우선)
                        json_user_id = data.get('user_id', file_owner)

                        # Handle both new and old JSON formats
                        if 'annotations' in data:
                            annotations_dict = data['annotations']
                        else:
                            annotations_dict = data

                        # Check each annotation for comments
                        for frame_key, frame_annotations in annotations_dict.items():
                            if frame_key in ['project_id', 'video_id', 'user_id']:
                                continue

                            if not isinstance(frame_annotations, list):
                                continue

                            for ann in frame_annotations:
                                if not isinstance(ann, dict):
                                    continue

                                comment = ann.get('comment', '').strip()
                                if comment:
                                    # created_by 결정: 어노테이션 > JSON user_id > file_owner
                                    # 이렇게 하면 annotation.json 같은 잘못된 파일명의 영향을 최소화
                                    created_by = ann.get('created_by') or json_user_id

                                    # 코멘트 고유 ID 생성 (json_user_id 사용)
                                    comment_id = generate_comment_id(
                                        user_owner,
                                        project_id,
                                        video_id,
                                        frame_key,
                                        json_user_id  # file_owner 대신 json_user_id 사용
                                    )

                                    # ⚡ 최적화: 이미 로드된 discussions_data 재사용
                                    replies_count = 0
                                    discussion_status = 'open'
                                    if comment_id in discussions_data['discussions']:
                                        replies_count = len(discussions_data['discussions'][comment_id]['replies'])
                                        discussion_status = discussions_data['discussions'][comment_id].get('status', 'open')

                                    # 프레임 이미지 URL 생성
                                    frame_url = f"/api/projects/{project_id}/videos/{video_id}/frame/{int(frame_key) if frame_key.isdigit() else frame_key}"

                                    # 댓글 작성자 정보 (modified_by가 있으면 그것을 사용, 없으면 어노테이션 작성자)
                                    comment_author = ann.get('modified_by') or created_by
                                    comment_author_name = ann.get('modified_by_name') or ann.get('created_by_name', comment_author)

                                    commented_annotations.append({
                                        'comment_id': comment_id,
                                        'project_owner': user_owner,
                                        'project_id': project_id,
                                        'project_name': project_name.rsplit('_', 1)[0],  # Remove timestamp
                                        'project_name_full': project_name,  # Full name with timestamp
                                        'video_id': video_id,
                                        'frame': int(frame_key) if frame_key.isdigit() else frame_key,
                                        'file_owner': file_owner,
                                        'created_by': comment_author,  # 댓글 작성자 (modified_by 우선)
                                        'created_by_name': comment_author_name,  # 댓글 작성자 이름
                                        'annotation_owner': created_by,  # 어노테이션 원 작성자
                                        'annotation_owner_name': ann.get('created_by_name', created_by),
                                        'label': ann.get('label', 'N/A'),
                                        'comment': comment,
                                        'bbox': ann.get('bbox'),
                                        'box': ann.get('box'),  # 바운딩 박스 데이터 추가
                                        'polygon': ann.get('polygon'),  # 다각형 데이터 추가
                                        'mask': ann.get('mask'),  # 마스크 데이터 추가
                                        'frame_url': frame_url,  # 프레임 이미지 URL 추가
                                        'created_at': ann.get('created_at'),
                                        'modified_at': ann.get('modified_at'),  # 댓글 수정 시간
                                        'replies_count': replies_count,
                                        'discussion_status': discussion_status
                                    })

        logger.info(f"[COMMENTS] Found {len(commented_annotations)} commented annotations for user {user_id}")
        return jsonify({
            'success': True,
            'comments': commented_annotations,
            'total': len(commented_annotations)
        })
    except Exception as e:
        logger.error(f"[COMMENTS] Error fetching commented annotations: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<path:project_id>/comments/counts', methods=['GET'])
@require_auth
def get_project_comment_counts(project_id: str):
    """프로젝트 내 비디오별 코멘트 개수 조회"""
    try:
        import json
        from pathlib import Path

        # 프로젝트 디렉토리 찾기
        project_dir = find_project_dir(project_id, request.user_id)
        if not project_dir:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 어노테이션 디렉토리
        annotations_dir = project_dir / 'annotations'
        if not annotations_dir.exists():
            return jsonify({
                'success': True,
                'counts': {}
            })

        # 비디오별 코멘트 개수 집계
        video_comment_counts = {}

        # 각 비디오 폴더 탐색
        for video_folder in annotations_dir.iterdir():
            if not video_folder.is_dir():
                continue

            video_id = video_folder.name
            comment_count = 0

            # 각 사용자의 어노테이션 파일 확인
            for json_file in video_folder.glob('*.json'):
                if 'backup' in json_file.name or 'before_fix' in json_file.name:
                    continue

                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    annotations = data.get('annotations', {})

                    # 코멘트가 있는 어노테이션 개수 세기
                    for frame_key, frame_annotations in annotations.items():
                        if isinstance(frame_annotations, list):
                            for ann in frame_annotations:
                                if isinstance(ann, dict) and ann.get('comment'):
                                    comment_count += 1

                except Exception as e:
                    logger.warning(f"[COMMENT_COUNTS] Error reading {json_file}: {e}")
                    continue

            if comment_count > 0:
                video_comment_counts[video_id] = comment_count

        logger.info(f"[COMMENT_COUNTS] Project {project_id}: {len(video_comment_counts)} videos with comments")
        return jsonify({
            'success': True,
            'counts': video_comment_counts
        })

    except Exception as e:
        logger.error(f"[COMMENT_COUNTS] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<path:project_id>/annotations/summary', methods=['GET'])
@require_auth
def get_annotation_summary(project_id: str):
    """프로젝트의 어노테이션 요약 정보 조회"""
    try:
        # 프로젝트 디렉토리 찾기
        project_dir = find_project_dir(project_id, request.user_id)
        if not project_dir:
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        # 프로젝트 정보 로드
        project_json = project_dir / 'project.json'
        if not project_json.exists():
            return jsonify({'success': False, 'error': 'Project metadata not found'}), 404

        with open(project_json, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        videos = project_data.get('videos', [])
        video_summary = {}  # Changed to dict for easier lookup by video_id

        # 프로젝트 전체 통계
        total_annotations = 0
        total_frames_set = set()
        by_class = {}
        by_worker = {}

        # 각 비디오의 어노테이션 통계 수집
        for video in videos:
            video_id = video.get('video_id')
            video_name = video.get('name', video_id)
            total_frames = video.get('total_frames', 0)

            # 비디오의 모든 어노테이션 파일 스캔
            annotated_frames = set()
            video_annotations = 0
            video_by_class = {}  # Per-video class statistics

            # annotations 폴더의 모든 사용자 JSON 파일 스캔
            annotations_dir = project_dir / 'annotations' / video_id
            if annotations_dir.exists():
                for json_file in annotations_dir.glob('*.json'):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        worker_name = json_file.stem

                        # 어노테이션 데이터 추출 (두 가지 형식 지원)
                        if isinstance(data, dict) and 'annotations' in data:
                            user_annotations = data['annotations']
                            worker_name = data.get('user_name', worker_name)
                        else:
                            user_annotations = data

                        for frame_key, frame_annos in user_annotations.items():
                            # frame_annos가 리스트인지 확인
                            if not isinstance(frame_annos, list):
                                continue

                            if frame_annos and len(frame_annos) > 0:
                                # 프레임 번호 추출
                                try:
                                    frame_num = int(frame_key)
                                    annotated_frames.add(frame_num)
                                    total_frames_set.add(f"{video_id}_{frame_num}")
                                except ValueError:
                                    continue

                                for anno in frame_annos:
                                    if not isinstance(anno, dict):
                                        continue

                                    video_annotations += 1
                                    total_annotations += 1

                                    # 클래스별 집계
                                    label = anno.get('label', '알 수 없음')
                                    by_class[label] = by_class.get(label, 0) + 1
                                    video_by_class[label] = video_by_class.get(label, 0) + 1

                                    # 작업자별 집계
                                    created_by_name = anno.get('created_by_name', worker_name)
                                    by_worker[created_by_name] = by_worker.get(created_by_name, 0) + 1

                    except Exception as e:
                        logger.warning(f"[ANNOTATION_SUMMARY] Error reading {json_file}: {e}")
                        continue

            video_summary[video_id] = {
                'name': video_name,
                'frames': len(annotated_frames),
                'total_annotations': video_annotations,
                'by_class': video_by_class
            }

        logger.info(f"[ANNOTATION_SUMMARY] Project {project_id}: {len(video_summary)} videos, {total_annotations} annotations")
        return jsonify({
            'success': True,
            'summary': {
                'videos': video_summary,
                'total_videos': len(videos),
                'total_annotations': total_annotations,
                'total_frames': len(total_frames_set),
                'by_class': by_class,
                'by_worker': by_worker
            }
        })

    except Exception as e:
        logger.error(f"[ANNOTATION_SUMMARY] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/annotations/global-summary', methods=['GET'])
@require_auth
def get_global_annotation_summary():
    """전체 어노테이션 요약 정보 조회 (모든 프로젝트)"""
    try:
        user_id = request.user_id

        # 사용자의 모든 프로젝트 찾기
        user_projects_dir = BASE_PROJECTS_DIR / user_id
        all_project_dirs = []

        # 본인 프로젝트
        if user_projects_dir.exists():
            all_project_dirs.extend([d for d in user_projects_dir.iterdir() if d.is_dir()])

        # 공유 프로젝트
        for owner_dir in BASE_PROJECTS_DIR.iterdir():
            if owner_dir.is_dir() and owner_dir.name != user_id:
                for project_dir in owner_dir.iterdir():
                    if project_dir.is_dir():
                        project_json = project_dir / 'project.json'
                        if project_json.exists():
                            try:
                                with open(project_json, 'r', encoding='utf-8') as f:
                                    project_data = json.load(f)
                                    if user_id in project_data.get('shared_with', []):
                                        all_project_dirs.append(project_dir)
                            except Exception as e:
                                logger.warning(f"[GLOBAL_SUMMARY] Error reading {project_json}: {e}")

        # 통계 집계
        total_projects = len(all_project_dirs)
        total_annotations = 0
        total_frames = set()
        by_class = {}
        by_worker = {}

        for project_dir in all_project_dirs:
            annotations_dir = project_dir / 'annotations'
            if not annotations_dir.exists():
                continue

            # 각 비디오 폴더 탐색
            for video_dir in annotations_dir.iterdir():
                if not video_dir.is_dir():
                    continue

                # 각 사용자의 어노테이션 파일 스캔
                for json_file in video_dir.glob('*.json'):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)

                        worker_name = json_file.stem  # 파일명이 작업자 ID

                        # 어노테이션 데이터 추출 (두 가지 형식 지원)
                        # 형식 1: {annotations: {frame_key: [annotations]}}
                        # 형식 2: {frame_key: [annotations]}
                        if isinstance(data, dict) and 'annotations' in data:
                            user_annotations = data['annotations']
                            worker_name = data.get('user_name', worker_name)
                        else:
                            user_annotations = data

                        for frame_key, frame_annos in user_annotations.items():
                            # frame_annos가 리스트인지 확인
                            if not isinstance(frame_annos, list):
                                continue

                            if frame_annos and len(frame_annos) > 0:
                                # 프레임 키 생성 (프로젝트/비디오/프레임 조합)
                                frame_id = f"{project_dir.name}/{video_dir.name}/{frame_key}"
                                total_frames.add(frame_id)

                                for anno in frame_annos:
                                    if not isinstance(anno, dict):
                                        continue

                                    total_annotations += 1

                                    # 클래스별 집계
                                    label = anno.get('label', '알 수 없음')
                                    by_class[label] = by_class.get(label, 0) + 1

                                    # 작업자별 집계
                                    created_by_name = anno.get('created_by_name', worker_name)
                                    by_worker[created_by_name] = by_worker.get(created_by_name, 0) + 1

                    except Exception as e:
                        logger.warning(f"[GLOBAL_SUMMARY] Error reading {json_file}: {e}")
                        continue

        logger.info(f"[GLOBAL_SUMMARY] User {user_id}: {total_projects} projects, {total_annotations} annotations")
        return jsonify({
            'success': True,
            'summary': {
                'total_projects': total_projects,
                'total_annotations': total_annotations,
                'total_frames': len(total_frames),
                'by_class': by_class,
                'by_worker': by_worker
            }
        })

    except Exception as e:
        logger.error(f"[GLOBAL_SUMMARY] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================
# 토론 스레드 기능
# ============================================

def generate_comment_id(project_owner, project_id, video_id, frame, annotation_owner):
    """코멘트 고유 ID 생성"""
    import hashlib
    data = f"{project_owner}_{project_id}_{video_id}_{frame}_{annotation_owner}"
    return hashlib.md5(data.encode()).hexdigest()[:16]


def get_discussions_file(project_owner, project_name):
    """토론 데이터 파일 경로 반환"""
    project_dir = BASE_PROJECTS_DIR / project_owner / project_name
    return project_dir / 'discussions.json'


def load_discussions(project_owner, project_name):
    """토론 데이터 로드"""
    discussions_file = get_discussions_file(project_owner, project_name)
    if discussions_file.exists():
        try:
            with open(discussions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[DISCUSSIONS] Error loading discussions: {e}")
            return {'discussions': {}}
    return {'discussions': {}}


def save_discussions(project_owner, project_name, data):
    """토론 데이터 저장"""
    discussions_file = get_discussions_file(project_owner, project_name)
    discussions_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(discussions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"[DISCUSSIONS] Error saving discussions: {e}")
        return False


@app.route('/api/discussions/reply', methods=['POST'])
@require_auth
def add_discussion_reply():
    """토론 답글 추가"""
    try:
        user_id = request.user_id
        data = request.json

        comment_id = data.get('comment_id')
        reply_text = data.get('reply', '').strip()
        project_owner = data.get('project_owner')
        project_name = data.get('project_name')

        if not all([comment_id, reply_text, project_owner, project_name]):
            return jsonify({
                'success': False,
                'error': '필수 필드가 누락되었습니다'
            }), 400

        # 토론 데이터 로드
        discussions_data = load_discussions(project_owner, project_name)

        # 토론 스레드가 없으면 생성
        if comment_id not in discussions_data['discussions']:
            discussions_data['discussions'][comment_id] = {
                'original_comment': data.get('original_comment', {}),
                'replies': [],
                'status': 'open'
            }

        # 답글 추가
        reply_id = f"{comment_id}_{len(discussions_data['discussions'][comment_id]['replies'])}_{int(time.time())}"
        new_reply = {
            'reply_id': reply_id,
            'user_id': user_id,
            'comment': reply_text,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        discussions_data['discussions'][comment_id]['replies'].append(new_reply)

        # 저장
        if save_discussions(project_owner, project_name, discussions_data):
            logger.info(f"[DISCUSSIONS] Reply added by {user_id} to comment {comment_id}")
            return jsonify({
                'success': True,
                'reply': new_reply,
                'total_replies': len(discussions_data['discussions'][comment_id]['replies'])
            })
        else:
            return jsonify({
                'success': False,
                'error': '답글 저장에 실패했습니다'
            }), 500

    except Exception as e:
        logger.error(f"[DISCUSSIONS] Error adding reply: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/discussions/reply', methods=['DELETE'])
@require_auth
def delete_discussion_reply():
    """토론 답글 삭제"""
    try:
        user_id = request.user_id
        data = request.json

        comment_id = data.get('comment_id')
        reply_index = data.get('reply_index')
        project_owner = data.get('project_owner')
        project_name = data.get('project_name')

        if not all([comment_id is not None, reply_index is not None, project_owner, project_name]):
            return jsonify({
                'success': False,
                'error': '필수 필드가 누락되었습니다'
            }), 400

        # 토론 데이터 로드
        discussions_data = load_discussions(project_owner, project_name)

        if comment_id not in discussions_data['discussions']:
            return jsonify({
                'success': False,
                'error': '토론을 찾을 수 없습니다'
            }), 404

        thread = discussions_data['discussions'][comment_id]
        replies = thread.get('replies', [])

        if reply_index >= len(replies):
            return jsonify({
                'success': False,
                'error': '답글을 찾을 수 없습니다'
            }), 404

        # 권한 확인: 본인이 작성한 답글만 삭제 가능
        if replies[reply_index].get('user_id') != user_id:
            return jsonify({
                'success': False,
                'error': '본인이 작성한 답글만 삭제할 수 있습니다'
            }), 403

        # 답글 삭제
        deleted_reply = replies.pop(reply_index)

        # 저장
        if save_discussions(project_owner, project_name, discussions_data):
            logger.info(f"[DISCUSSIONS] Reply deleted by {user_id} from comment {comment_id}")
            return jsonify({
                'success': True,
                'total_replies': len(replies)
            })
        else:
            return jsonify({
                'success': False,
                'error': '답글 삭제에 실패했습니다'
            }), 500

    except Exception as e:
        logger.error(f"[DISCUSSIONS] Error deleting reply: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/discussions/reply', methods=['PUT'])
@require_auth
def update_discussion_reply():
    """토론 답글 수정"""
    try:
        user_id = request.user_id
        data = request.json

        comment_id = data.get('comment_id')
        reply_index = data.get('reply_index')
        new_comment = data.get('new_comment', '').strip()
        project_owner = data.get('project_owner')
        project_name = data.get('project_name')

        if not all([comment_id is not None, reply_index is not None, new_comment, project_owner, project_name]):
            return jsonify({
                'success': False,
                'error': '필수 필드가 누락되었습니다'
            }), 400

        # 토론 데이터 로드
        discussions_data = load_discussions(project_owner, project_name)

        if comment_id not in discussions_data['discussions']:
            return jsonify({
                'success': False,
                'error': '토론을 찾을 수 없습니다'
            }), 404

        thread = discussions_data['discussions'][comment_id]
        replies = thread.get('replies', [])

        if reply_index >= len(replies):
            return jsonify({
                'success': False,
                'error': '답글을 찾을 수 없습니다'
            }), 404

        # 권한 확인: 본인이 작성한 답글만 수정 가능
        if replies[reply_index].get('user_id') != user_id:
            return jsonify({
                'success': False,
                'error': '본인이 작성한 답글만 수정할 수 있습니다'
            }), 403

        # 답글 수정
        replies[reply_index]['comment'] = new_comment
        replies[reply_index]['updated_at'] = datetime.now().isoformat()

        # 저장
        if save_discussions(project_owner, project_name, discussions_data):
            logger.info(f"[DISCUSSIONS] Reply updated by {user_id} in comment {comment_id}")
            return jsonify({
                'success': True,
                'reply': replies[reply_index]
            })
        else:
            return jsonify({
                'success': False,
                'error': '답글 수정에 실패했습니다'
            }), 500

    except Exception as e:
        logger.error(f"[DISCUSSIONS] Error updating reply: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/discussions/<comment_id>', methods=['GET'])
@require_auth
def get_discussion_thread(comment_id):
    """특정 토론 스레드 조회"""
    try:
        project_owner = request.args.get('project_owner')
        project_name = request.args.get('project_name')

        if not all([project_owner, project_name]):
            return jsonify({
                'success': False,
                'error': '프로젝트 정보가 필요합니다'
            }), 400

        discussions_data = load_discussions(project_owner, project_name)
        thread = discussions_data['discussions'].get(comment_id)

        if thread:
            return jsonify({
                'success': True,
                'thread': thread
            })
        else:
            return jsonify({
                'success': True,
                'thread': {
                    'original_comment': {},
                    'replies': [],
                    'status': 'open'
                }
            })

    except Exception as e:
        logger.error(f"[DISCUSSIONS] Error fetching thread: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/discussions/<comment_id>/status', methods=['PUT'])
@require_auth
def update_discussion_status(comment_id):
    """토론 상태 업데이트"""
    try:
        data = request.json
        project_owner = data.get('project_owner')
        project_name = data.get('project_name')
        new_status = data.get('status')

        if not all([project_owner, project_name, new_status]):
            return jsonify({
                'success': False,
                'error': '필수 필드가 누락되었습니다'
            }), 400

        if new_status not in ['open', 'resolved', 'pending']:
            return jsonify({
                'success': False,
                'error': '유효하지 않은 상태입니다'
            }), 400

        discussions_data = load_discussions(project_owner, project_name)

        if comment_id in discussions_data['discussions']:
            discussions_data['discussions'][comment_id]['status'] = new_status

            if save_discussions(project_owner, project_name, discussions_data):
                logger.info(f"[DISCUSSIONS] Status updated for {comment_id} to {new_status}")
                return jsonify({
                    'success': True,
                    'status': new_status
                })

        return jsonify({
            'success': False,
            'error': '토론 스레드를 찾을 수 없습니다'
        }), 404

    except Exception as e:
        logger.error(f"[DISCUSSIONS] Error updating status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test', methods=['GET'])
def test():
    """테스트 엔드포인트"""
    import sys
    return jsonify({
        'message': 'Proxy Backend (Forwarding to GPU Server)',
        'python_version': sys.version,
        'gpu_server': GPU_SERVER_URL
    })


@app.route('/')
def index():
    """메인 페이지"""
    return send_from_directory('.', 'index.html')


@app.route('/api/inference/status/<job_id>', methods=['GET'])
def get_inference_status(job_id):
    """추론 작업 상태 조회 (GPU 서버로 전달)"""
    try:
        response = requests.get(
            f"{GPU_SERVER_URL}/api/inference/status/{job_id}",
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"[INFERENCE] Status check error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/cancel/<job_id>', methods=['POST'])
def cancel_inference(job_id):
    """추론 작업 취소 (GPU 서버로 전달)"""
    try:
        response = requests.post(
            f"{GPU_SERVER_URL}/api/inference/cancel/{job_id}",
            timeout=5
        )
        logger.info(f"[INFERENCE] Cancel request for job {job_id}")
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"[INFERENCE] Cancel error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/preview/<job_id>', methods=['GET'])
def get_inference_preview(job_id):
    """추론 작업의 최신 미리보기 프레임 가져오기 (GPU 서버로 전달)"""
    try:
        response = requests.get(f"{GPU_SERVER_URL}/api/inference/preview/{job_id}", timeout=5, stream=True)

        if response.status_code == 200:
            # 이미지를 스트림으로 전달
            return Response(response.iter_content(chunk_size=8192),
                          mimetype='image/jpeg',
                          headers={'Cache-Control': 'no-cache'})
        else:
            return jsonify(response.json()), response.status_code

    except Exception as e:
        logger.error(f"[INFERENCE] Preview error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/frames/<job_id>', methods=['GET'])
def get_inference_frames(job_id):
    """추론 작업의 프레임 정보 가져오기 (GPU 서버로 전달)"""
    try:
        response = requests.get(f"{GPU_SERVER_URL}/api/inference/frames/{job_id}", timeout=5)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"[INFERENCE] Frames info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/frame/<job_id>/<int:frame_index>', methods=['GET'])
def get_inference_frame(job_id, frame_index):
    """특정 프레임 이미지 가져오기 (GPU 서버로 전달)"""
    try:
        response = requests.get(f"{GPU_SERVER_URL}/api/inference/frame/{job_id}/{frame_index}", timeout=5, stream=True)

        if response.status_code == 200:
            # 이미지를 스트림으로 전달
            return Response(response.iter_content(chunk_size=8192),
                          mimetype='image/jpeg',
                          headers={'Cache-Control': 'no-cache'})
        else:
            return jsonify(response.json()), response.status_code

    except Exception as e:
        logger.error(f"[INFERENCE] Frame {frame_index} error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/check', methods=['POST'])
def check_inference_results():
    """추론 결과 존재 여부 확인 (GPU 서버로 전달)"""
    try:
        data = request.get_json()
        response = requests.post(f"{GPU_SERVER_URL}/api/inference/check", json=data, timeout=5)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"[INFERENCE] Check error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/completed-frame', methods=['POST'])
def get_completed_frame():
    """완료된 추론 결과 프레임 가져오기 (GPU 서버로 전달)"""
    try:
        data = request.get_json()
        response = requests.post(
            f"{GPU_SERVER_URL}/api/inference/completed-frame",
            json=data,
            timeout=5,
            stream=True
        )

        if response.status_code == 200:
            return Response(
                response.iter_content(chunk_size=8192),
                mimetype='image/jpeg',
                headers={'Cache-Control': 'public, max-age=3600'}
            )
        else:
            return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"[INFERENCE] Completed frame error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference', methods=['POST'])
def run_video_inference():
    """비디오 추론 실행 (GPU 서버로 전달)"""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        model_path = data.get('model_path')
        video_path = data.get('video_path')
        output_path = data.get('output_path', 'inference_results')

        logger.info(f"[INFERENCE] Request received - Model: {model_type}, Video: {video_path}")
        logger.info(f"[INFERENCE] Forwarding to GPU server: {GPU_SERVER_URL}/api/inference")

        # GPU 서버로 요청 전달
        response = requests.post(
            f"{GPU_SERVER_URL}/api/inference",
            json={
                'model_type': model_type,
                'model_path': model_path,
                'video_path': video_path,
                'output_path': output_path
            },
            timeout=3600  # 1시간 타임아웃 (긴 비디오 추론 대응)
        )

        logger.info(f"[INFERENCE] GPU server responded with status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            logger.info(f"[INFERENCE] Success - Output: {result.get('output_dir', 'N/A')}")
            return jsonify(result)
        else:
            error_msg = f"GPU server error: {response.status_code}"
            logger.error(f"[INFERENCE] {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), response.status_code

    except requests.exceptions.Timeout:
        logger.error("[INFERENCE] Timeout - inference took too long")
        return jsonify({
            'success': False,
            'error': 'Inference timeout (>1 hour)'
        }), 504
    except requests.exceptions.RequestException as e:
        logger.error(f"[INFERENCE] Request error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to connect to GPU server: {str(e)}'
        }), 503
    except Exception as e:
        logger.error(f"[INFERENCE] Unexpected error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Temporarily disabled catch-all route to fix frame image loading
# @app.route('/<path:path>')
# def serve_static(path):
#     """정적 파일 서빙"""
#     return send_from_directory('.', path)


if __name__ == '__main__':
    import threading
    import time

    def cleanup_sessions():
        """주기적으로 만료된 세션 정리"""
        while True:
            time.sleep(600)  # 10분마다 실행
            count = user_manager.cleanup_expired_sessions()
            if count > 0:
                logger.info(f"[AUTH] Cleaned up {count} expired sessions")
@app.route('/api/admin/dashboard', methods=['GET'])
@require_auth
def get_admin_dashboard():
    """관리자용 전체 프로젝트 통계 대시보드"""
    import json
    from pathlib import Path
    from collections import defaultdict

    try:
        # 관리자 권한 확인
        if request.user_id != 'admin':
            return jsonify({
                'success': False,
                'error': 'Admin access required'
            }), 403

        # 모든 사용자의 프로젝트 스캔
        all_projects = []

        for user_dir in BASE_PROJECTS_DIR.iterdir():
            if not user_dir.is_dir():
                continue

            user_id = user_dir.name

            # 사용자 정보 가져오기
            user_info = user_manager.get_user_info(user_id)
            user_name = user_info.get('full_name', user_id) if user_info else user_id

            # 사용자의 모든 프로젝트 스캔
            for project_dir in user_dir.iterdir():
                if not project_dir.is_dir():
                    continue

                project_file = project_dir / 'project.json'
                if not project_file.exists():
                    continue

                try:
                    with open(project_file, 'r', encoding='utf-8') as f:
                        project_data = json.load(f)

                    project_id = project_data.get('id', project_dir.name)
                    project_name = project_data.get('name', 'Unnamed Project')
                    classes = project_data.get('classes', [])
                    videos = project_data.get('videos', [])

                    # 어노테이션 통계 계산
                    annotations_dir = project_dir / 'annotations'
                    total_annotations = 0
                    annotated_frames = 0
                    class_distribution = defaultdict(int)
                    contributors = set()
                    video_frame_counts = {}

                    if annotations_dir.exists():
                        for video in videos:
                            video_id = video.get('video_id')
                            if not video_id:
                                continue

                            video_annotations_dir = annotations_dir / video_id
                            if not video_annotations_dir.exists():
                                continue

                            video_frames = set()
                            video_ann_count = 0

                            # 모든 사용자의 어노테이션 파일 읽기
                            for ann_file in video_annotations_dir.glob('*.json'):
                                if ann_file.stem.endswith('.backup'):
                                    continue

                                try:
                                    with open(ann_file, 'r', encoding='utf-8') as af:
                                        ann_data = json.load(af)

                                    contributor_id = ann_data.get('user_id', ann_file.stem)
                                    contributors.add(contributor_id)

                                    annotations = ann_data.get('annotations', {})
                                    for frame_key, frame_anns in annotations.items():
                                        if not isinstance(frame_anns, list) or len(frame_anns) == 0:
                                            continue

                                        video_frames.add(frame_key)

                                        for ann in frame_anns:
                                            if isinstance(ann, dict):
                                                video_ann_count += 1
                                                total_annotations += 1
                                                label = ann.get('label', 'unknown')
                                                class_distribution[label] += 1

                                except Exception as e:
                                    logger.warning(f"[ADMIN DASHBOARD] Failed to read {ann_file}: {e}")

                            if len(video_frames) > 0:
                                annotated_frames += len(video_frames)
                                video_frame_counts[video_id] = {
                                    'frames': len(video_frames),
                                    'annotations': video_ann_count
                                }

                    # 비디오 통계
                    total_videos = len(videos)
                    completed_videos = sum(1 for v in videos if v.get('status') == 'completed')

                    all_projects.append({
                        'user_id': user_id,
                        'user_name': user_name,
                        'project_id': project_id,
                        'project_name': project_name,
                        'classes': classes,
                        'total_videos': total_videos,
                        'completed_videos': completed_videos,
                        'total_annotations': total_annotations,
                        'annotated_frames': annotated_frames,
                        'class_distribution': dict(class_distribution),
                        'contributors': list(contributors),
                        'video_details': video_frame_counts,
                        'created_at': project_data.get('created', ''),
                        'updated_at': project_data.get('updated', '')
                    })

                except Exception as e:
                    logger.warning(f"[ADMIN DASHBOARD] Failed to process project {project_dir}: {e}")

        # 전체 통계
        total_stats = {
            'total_projects': len(all_projects),
            'total_videos': sum(p['total_videos'] for p in all_projects),
            'completed_videos': sum(p['completed_videos'] for p in all_projects),
            'total_annotations': sum(p['total_annotations'] for p in all_projects),
            'annotated_frames': sum(p['annotated_frames'] for p in all_projects),
            'unique_contributors': len(set(c for p in all_projects for c in p['contributors']))
        }

        logger.info(f"[ADMIN DASHBOARD] Fetched {len(all_projects)} projects for admin")

        return jsonify({
            'success': True,
            'projects': all_projects,
            'summary': total_stats
        })

    except Exception as e:
        logger.error(f"[ADMIN DASHBOARD] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # 세션 정리 스레드 시작
    cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
    cleanup_thread.start()

    logger.info("🚀 Starting Client Backend (Proxy Mode)...")
    logger.info(f"📡 API Server: http://localhost:5001")
    logger.info(f"🔌 GPU Server: {GPU_SERVER_URL}")
    logger.info(f"👥 User Management: Enabled")
    logger.info(f"🔐 Default Admin: admin/admin123")

    # 등록된 라우트 목록 출력 (디버깅)
    logger.info("📋 Registered routes:")
    for rule in app.url_map.iter_rules():
        logger.info(f"  {rule.endpoint}: {rule.rule} [{', '.join(rule.methods - {'HEAD', 'OPTIONS'})}]")

    # debug=False로 변경하여 reloader 문제 방지
    app.run(host='0.0.0.0', port=5003, debug=False, threaded=True)
