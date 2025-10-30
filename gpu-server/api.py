#!/usr/bin/env python3
"""
GPU Server API
Grounded-SAM 작업을 수행하는 GPU 서버용 REST API
"""

from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
import sys
import os
import cv2
import numpy as np
from io import BytesIO
import torch
from PIL import Image
import base64
import subprocess
import math
import threading
import json

# Grounded-SAM 경로 추가
# sys.path.insert(0, '/home/ppak/SynologyDrive/ykpark/linux_devel/ground_sam/Grounded-Segment-Anything')

from project_manager import ProjectManager

# SegFormer 모델 전역 변수
segformer_model = None
segformer_processor = None
segformer_device = None
ai_initialized = False

# 추론 락 (멀티스레드 환경에서 동시 추론 방지)
inference_lock = threading.Lock()
inference_stats = {
    'total_requests': 0,
    'active_requests': 0,
    'max_concurrent': 0
}

# 작업 관리 (진행 상황 추적 및 취소)
active_jobs = {}  # job_id -> { 'status', 'progress', 'cancel_requested', 'video_path', ... }
job_lock = threading.Lock()

app = Flask(__name__)
CORS(app)


@app.route('/api/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'ok',
        'message': 'GPU Server is running',
        'gpu_available': True,  # TODO: 실제 GPU 확인 로직
        'threading': 'enabled',
        'inference_stats': {
            'total_requests': inference_stats['total_requests'],
            'active_requests': inference_stats['active_requests'],
            'max_concurrent': inference_stats['max_concurrent']
        }
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """서버 통계 조회"""
    import psutil

    return jsonify({
        'success': True,
        'inference': {
            'total_requests': inference_stats['total_requests'],
            'active_requests': inference_stats['active_requests'],
            'max_concurrent': inference_stats['max_concurrent']
        },
        'server': {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'threads': threading.active_count()
        },
        'ai_model': {
            'initialized': ai_initialized,
            'device': str(segformer_device) if segformer_device else None
        }
    })


@app.route('/api/projects', methods=['GET'])
def list_projects():
    """프로젝트 목록 조회"""
    try:
        pm = ProjectManager()
        projects = pm.list_projects()

        projects_data = []
        for p in projects:
            projects_data.append({
                'id': p.id,
                'name': p.name,
                'path': str(p.project_dir),
                'classes': p.classes
            })

        return jsonify({
            'success': True,
            'projects': projects_data
        })
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects', methods=['POST'])
def create_project():
    """프로젝트 생성"""
    try:
        pm = ProjectManager()
        data = request.json

        classes = data['classes']
        if isinstance(classes, str):
            classes = [c.strip() for c in classes.split(',')]

        project = pm.create_project(
            name=data['name'],
            classes=classes,
            description=data.get('description', '')
        )

        return jsonify({
            'success': True,
            'project': {
                'id': project.id,
                'name': project.name,
                'path': str(project.project_dir)
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """프로젝트 상세 정보"""
    try:
        pm = ProjectManager()
        projects = pm.list_projects()

        project = None
        for p in projects:
            if p.id == project_id:
                project = p
                break

        if not project:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        stats = project.get_statistics()
        videos_data = []
        for video in project.videos:
            videos_data.append({
                'id': video.get('video_id', ''),
                'filename': video.get('filename', ''),
                'total_frames': video.get('total_frames', 0)
            })

        return jsonify({
            'success': True,
            'project': {
                'id': project.id,
                'name': project.name,
                'path': str(project.project_dir),
                'classes': project.classes,
                'stats': stats,
                'videos': videos_data
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos', methods=['POST'])
def add_video(project_id):
    """비디오 추가 (Backend Proxy에서 파일 업로드 완료 후 호출 또는 NAS 비디오 참조)"""
    try:
        from pathlib import Path
        import json
        from datetime import datetime

        data = request.json
        print(f"[DEBUG] Received data: {data}", flush=True)

        project_dir = data.get('project_dir')
        if not project_dir:
            return jsonify({'success': False, 'error': 'project_dir required'}), 400

        # NAS 비디오 참조인지 확인
        is_nas_reference = data.get('is_nas_reference', False)

        if is_nas_reference:
            # NAS 비디오 참조 처리
            nas_video_path = data.get('nas_video_path')
            if not nas_video_path:
                return jsonify({'success': False, 'error': 'nas_video_path required for NAS reference'}), 400

            video_path = nas_video_path
            nas_metadata = data.get('nas_metadata', {})
            print(f"[NAS] Adding NAS video reference: {nas_video_path}", flush=True)
        else:
            # 일반 업로드된 비디오 처리
            video_path = data.get('video_path')
            if not video_path:
                return jsonify({'success': False, 'error': 'video_path required'}), 400

            parent_dir = data.get('parent_dir', '')  # 부모 디렉토리명 (선택사항)

        print(f"[DEBUG] Video path: {video_path}", flush=True)
        print(f"[DEBUG] Project dir: {project_dir}", flush=True)

        # project.json 파일 경로
        project_json_path = Path(project_dir) / 'project.json'

        if not project_json_path.exists():
            print(f"[ERROR] Project file not found: {project_json_path}", flush=True)
            return jsonify({'success': False, 'error': 'Project file not found'}), 404

        # project.json 읽기
        with open(project_json_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 비디오 ID (backend proxy에서 전달받거나 새로 생성)
        video_id = data.get('video_id')
        if not video_id:
            import time
            video_id = f"video_{int(time.time() * 1000)}"

        # 비디오 프레임 수 및 해상도 계산
        import cv2
        total_frames = 0
        width = 0
        height = 0
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"[DEBUG] Video info: {total_frames} frames, {width}x{height}", flush=True)
        except Exception as e:
            print(f"[WARNING] Could not get video info: {e}", flush=True)

        # 비디오 정보 추가
        if 'videos' not in project_data:
            project_data['videos'] = []

        video_info = {
            'video_id': video_id,
            'filename': Path(video_path).name,
            'video_path': str(video_path),
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'added_at': datetime.now().isoformat()
        }

        # NAS 비디오 참조인 경우 추가 정보
        if is_nas_reference:
            video_info['is_nas_reference'] = True
            video_info['nas_metadata'] = nas_metadata
            print(f"[NAS] NAS metadata: {nas_metadata}", flush=True)
        else:
            # 일반 업로드 비디오 - 부모 디렉토리명 추가
            if 'parent_dir' in locals() and parent_dir:
                video_info['parent_dir'] = parent_dir
                print(f"[DEBUG] Parent dir: {parent_dir}", flush=True)

        project_data['videos'].append(video_info)

        # project.json 저장
        with open(project_json_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Video added: {video_id} (NAS reference: {is_nas_reference})", flush=True)

        return jsonify({
            'success': True,
            'video_id': video_id
        })

    except Exception as e:
        print(f"[ERROR] Add video error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>', methods=['GET'])
def get_video(project_id, video_id):
    """비디오 상세 정보 조회"""
    try:
        from pathlib import Path
        import json
        from flask import request

        project_dir = request.args.get('project_dir')
        if not project_dir:
            return jsonify({'success': False, 'error': 'project_dir parameter required'}), 400

        print(f"[DEBUG] GET Video - Project dir: {project_dir}, video_id: {video_id}", flush=True)

        # project.json 파일 경로
        project_json_path = Path(project_dir) / 'project.json'

        if not project_json_path.exists():
            print(f"[ERROR] Project file not found: {project_json_path}", flush=True)
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        # project.json 읽기
        with open(project_json_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 프로젝트의 비디오 목록에서 해당 비디오 찾기
        video_info = None
        for video in project_data.get('videos', []):
            if video.get('video_id') == video_id:
                video_info = video
                break

        if not video_info:
            print(f"[ERROR] Video not found: {video_id}", flush=True)
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        # 비디오 파일 경로
        video_path = video_info.get('video_path', str(Path(project_dir) / 'videos' / video_info['filename']))

        print(f"[SUCCESS] Video found: {video_id}", flush=True)

        return jsonify({
            'success': True,
            'video': {
                'id': video_info['video_id'],
                'filename': video_info['filename'],
                'path': video_path,
                'total_frames': video_info.get('total_frames', 0),
                'frame_count': video_info.get('total_frames', 0),
                'annotations': video_info.get('frame_count', 0),
                'status': video_info.get('status', 'in_progress'),  # 비디오 상태 추가
                'nas_metadata': video_info.get('nas_metadata')  # NAS 메타데이터도 포함
            }
        })
    except Exception as e:
        print(f"[ERROR] Get video error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>/stream', methods=['GET'])
def stream_video(project_id, video_id):
    """비디오 파일 스트리밍"""
    try:
        pm = ProjectManager()
        projects = pm.list_projects()

        project = None
        for p in projects:
            if p.id == project_id:
                project = p
                break

        if not project:
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        # 프로젝트의 비디오 목록에서 해당 비디오 찾기
        video_info = None
        for video in project.videos:
            if video.get('video_id') == video_id:
                video_info = video
                break

        if not video_info:
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        # 비디오 파일 경로 구성 (절대 경로로 해결)
        from pathlib import Path
        video_path = str((Path(project.project_dir) / 'videos' / video_info['filename']).resolve())

        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': f'Video file not found: {video_path}'}), 404

        # 파일 확장자에 따라 mimetype 설정
        import mimetypes
        mimetype, _ = mimetypes.guess_type(video_path)
        if not mimetype:
            # 확장자에 따라 기본 mimetype 설정
            ext = os.path.splitext(video_path)[1].lower()
            mimetype_map = {
                '.mp4': 'video/mp4',
                '.avi': 'video/x-msvideo',
                '.mov': 'video/quicktime',
                '.mkv': 'video/x-matroska',
                '.webm': 'video/webm'
            }
            mimetype = mimetype_map.get(ext, 'video/mp4')

        return send_file(video_path, mimetype=mimetype)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>/frame/<int:frame_number>', methods=['GET'])
def get_video_frame(project_id, video_id, frame_number):
    """비디오의 특정 프레임을 이미지로 추출"""
    try:
        from pathlib import Path
        import json
        from flask import request

        project_dir = request.args.get('project_dir')
        if not project_dir:
            return jsonify({'success': False, 'error': 'project_dir parameter required'}), 400

        print(f"[DEBUG] GET Frame - Project dir: {project_dir}, video_id: {video_id}, frame: {frame_number}", flush=True)

        # project.json 파일 경로
        project_json_path = Path(project_dir) / 'project.json'

        if not project_json_path.exists():
            print(f"[ERROR] Project file not found: {project_json_path}", flush=True)
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        # project.json 읽기
        with open(project_json_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 프로젝트의 비디오 목록에서 해당 비디오 찾기
        video_info = None
        for video in project_data.get('videos', []):
            if video.get('video_id') == video_id:
                video_info = video
                break

        if not video_info:
            print(f"[ERROR] Video not found: {video_id}", flush=True)
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        # 비디오 파일 경로
        video_path = video_info.get('video_path', str(Path(project_dir) / 'videos' / video_info['filename']))

        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': f'Video file not found: {video_path}'}), 404

        # OpenCV로 비디오 열기 (기본 백엔드 사용 - 가장 안정적)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Failed to open video'}), 500

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 프레임 범위 검증
        if frame_number < 0 or frame_number >= total_frames:
            cap.release()
            return jsonify({
                'success': False,
                'error': f'Frame {frame_number} out of range (0-{total_frames-1})'
            }), 400

        # 프레임 읽기 (기존 pipe_video_inspector.py와 동일한 방식)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        # 프레임 읽기 실패 시 - 초기 프레임이면 자동으로 유효한 프레임 찾기
        if not ret or frame is None:
            if frame_number < 200:  # 초기 200 프레임 내에서만 fallback 시도
                print(f"[INFO] Frame {frame_number} failed, searching for first valid frame...")
                # 10, 30, 50, 100, 150, 200 순서로 시도
                for fallback_frame in [10, 30, 50, 100, 150, 200]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"[INFO] Using frame {fallback_frame} as fallback for frame {frame_number}")
                        break

        cap.release()

        # 최종 실패
        if not ret or frame is None:
            return jsonify({
                'success': False,
                'error': f'Failed to read frame {frame_number}. This video may have corrupted frames.'
            }), 400

        # JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # BytesIO로 변환
        img_io = BytesIO(buffer.tobytes())
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>', methods=['DELETE'])
def remove_video(project_id, video_id):
    """비디오 제거 (NAS 참조는 파일 삭제 안함)"""
    try:
        from pathlib import Path
        import json
        import os
        from flask import request

        project_dir = request.args.get('project_dir')
        if not project_dir:
            return jsonify({'success': False, 'error': 'project_dir parameter required'}), 400

        print(f"[DEBUG] DELETE Video - Project dir: {project_dir}, video_id: {video_id}", flush=True)

        # project.json 파일 경로
        project_json_path = Path(project_dir) / 'project.json'

        if not project_json_path.exists():
            print(f"[ERROR] Project file not found: {project_json_path}", flush=True)
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        # project.json 읽기
        with open(project_json_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 비디오 찾기 및 제거
        video_info = None
        videos = project_data.get('videos', [])
        for i, video in enumerate(videos):
            if video.get('video_id') == video_id:
                video_info = videos.pop(i)
                break

        if not video_info:
            print(f"[ERROR] Video not found: {video_id}", flush=True)
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        # NAS 참조가 아닌 경우에만 비디오 파일 삭제
        is_nas_reference = video_info.get('is_nas_reference', False)
        if not is_nas_reference:
            video_path = video_info.get('video_path')
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"[DEBUG] Video file deleted: {video_path}", flush=True)
        else:
            print(f"[NAS] Skipping file deletion for NAS reference: {video_info.get('video_path')}", flush=True)

        # project.json 저장
        with open(project_json_path, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        print(f"[SUCCESS] Video removed: {video_id} (NAS reference: {is_nas_reference})", flush=True)

        return jsonify({'success': True})
    except Exception as e:
        print(f"[ERROR] Remove video error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def load_ai_model():
    """SegFormer 모델 로드 (서버 시작 시 자동 실행)"""
    global segformer_model, segformer_processor, segformer_device, ai_initialized

    try:
        print("[AI] Initializing custom SegFormer model...")

        # 디바이스 설정
        if torch.cuda.is_available():
            segformer_device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            print(f"[AI] Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            segformer_device = torch.device("cpu")
            print("[AI] Using CPU (GPU not available)")

        # 커스텀 SegFormer 모델 로드
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

        # 모델 체크포인트 경로 확인
        # 프로젝트 루트 디렉토리 기준으로 경로 설정
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(script_dir, 'segformer_best.pth')

        if not os.path.exists(model_path):
            print(f"[AI] Warning: Custom model not found at {model_path}")
            print("[AI] Using pretrained model instead")
            model_path = None
            base_model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
        else:
            print(f"[AI] Loading custom model from: {model_path}")

            # 체크포인트에서 모델 아키텍처 정보 읽기 (pipe_video_inspector.py와 동일)
            checkpoint = torch.load(model_path, map_location=segformer_device)
            if isinstance(checkpoint, dict) and 'model_name' in checkpoint:
                base_model_name = checkpoint['model_name']
                print(f"[AI] Using model architecture from checkpoint: {base_model_name}")
            else:
                base_model_name = "nvidia/segformer-b3-finetuned-ade-512-512"
                print(f"[AI] Using default architecture: {base_model_name}")

        # 프로세서 초기화 (pipe_video_inspector.py와 동일하게 기본 생성자 사용)
        segformer_processor = SegformerImageProcessor()

        # 모델 아키텍처 로드
        segformer_model = SegformerForSemanticSegmentation.from_pretrained(
            base_model_name,
            num_labels=3,  # rust, scale, background
            ignore_mismatched_sizes=True
        )

        model_info = 'pretrained (nvidia/segformer-b3)'

        # 커스텀 가중치 로드
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=segformer_device)

                # state_dict 추출
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint

                # 모델에 가중치 로드
                segformer_model.load_state_dict(state_dict, strict=False)
                model_info = 'segformer_best.pth (custom trained)'
                print("[AI] Custom weights loaded successfully")
            except Exception as e:
                print(f"[AI] Warning: Failed to load custom weights: {e}")
                print("[AI] Continuing with pretrained weights")

        segformer_model.to(segformer_device)
        segformer_model.eval()

        ai_initialized = True
        print(f"[AI] SegFormer model initialized successfully ({model_info})")
        return True

    except Exception as e:
        print(f"[AI] Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/api/ai/initialize', methods=['POST'])
def initialize_ai():
    """SegFormer 모델 초기화 (API 엔드포인트 - 레거시 호환용)"""
    global ai_initialized

    if ai_initialized:
        return jsonify({
            'success': True,
            'message': 'AI model already initialized',
            'device': str(segformer_device)
        })

    success = load_ai_model()

    if success:
        return jsonify({
            'success': True,
            'message': 'AI model initialized',
            'device': str(segformer_device)
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to initialize AI model'
        }), 500


def extract_bounding_boxes_from_mask(mask, min_area=100, include_masks=False):
    """세그멘테이션 마스크에서 바운딩 박스 추출"""
    import cv2

    boxes = []
    unique_classes = np.unique(mask)

    # 클래스 이름 매핑
    class_names = {
        0: 'background',
        1: 'rust',
        2: 'scale'
    }

    # 배경(0) 제외
    for class_id in unique_classes:
        if class_id == 0:
            continue

        # 해당 클래스의 마스크 생성
        class_mask = (mask == class_id).astype(np.uint8) * 255

        # 컨투어 찾기
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            # 바운딩 박스 추출
            x, y, w, h = cv2.boundingRect(contour)

            box_data = {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'label': class_names.get(int(class_id), f'class_{class_id}'),
                'class_id': int(class_id),
                'area': float(area),
                'confidence': float(area / (w * h)) if w * h > 0 else 0.0  # 박스 내 클래스 픽셀 비율
            }

            # 마스크 포함 옵션
            if include_masks:
                # 바운딩 박스 영역의 마스크 추출 (해당 클래스만)
                cropped_mask = (mask[y:y+h, x:x+w] == class_id).astype(np.uint8)

                # PNG로 인코딩
                mask_png = Image.fromarray(cropped_mask, mode='L')
                mask_buffer = BytesIO()
                mask_png.save(mask_buffer, format='PNG')
                box_data['mask'] = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')

                # 폴리곤 추출 (윤곽선 단순화)
                epsilon = 0.005 * cv2.arcLength(contour, True)  # 0.5% 단순화
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)

                # 폴리곤 포인트를 리스트로 변환
                polygon_points = []
                for point in approx_contour:
                    polygon_points.append({
                        'x': int(point[0][0]),
                        'y': int(point[0][1])
                    })

                box_data['polygon'] = polygon_points
                print(f"[POLYGON] Extracted {len(polygon_points)} points for {class_names.get(int(class_id))}")

            boxes.append(box_data)

    return boxes


@app.route('/api/ai/inference', methods=['POST'])
def run_inference():
    """현재 프레임에 대해 SegFormer 추론 실행 및 바운딩 박스 추출"""
    global segformer_model, segformer_processor, segformer_device, ai_initialized, inference_stats

    # 추론 락 획득 (순차 처리)
    with inference_lock:
        inference_stats['total_requests'] += 1
        inference_stats['active_requests'] += 1
        inference_stats['max_concurrent'] = max(inference_stats['max_concurrent'], inference_stats['active_requests'])

        try:
            return _run_inference_internal()
        finally:
            inference_stats['active_requests'] -= 1

def _run_inference_internal():
    """내부 추론 로직 (락으로 보호됨)"""
    global segformer_model, segformer_processor, segformer_device, ai_initialized

    try:
        if not ai_initialized or segformer_model is None:
            return jsonify({
                'success': False,
                'error': 'AI model not initialized. Call /api/ai/initialize first.'
            }), 400

        data = request.json
        project_dir = data.get('project_dir')
        video_id = data.get('video_id')
        frame_number = data.get('frame_number')

        if not all([project_dir, video_id, frame_number is not None]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: project_dir, video_id, frame_number'
            }), 400

        print(f"[AI] Running inference on frame {frame_number}")

        # project.json에서 직접 프로젝트 정보 읽기
        from pathlib import Path
        import json

        project_json_path = Path(project_dir) / 'project.json'
        if not project_json_path.exists():
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        with open(project_json_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 비디오 정보 찾기
        video_info = None
        for video in project_data.get('videos', []):
            if video.get('video_id') == video_id:
                video_info = video
                break

        if not video_info:
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        video_path = video_info.get('video_path')
        if not video_path or not os.path.exists(video_path):
            return jsonify({'success': False, 'error': f'Video file not found: {video_path}'}), 404

        # 프레임 추출
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Failed to open video'}), 500

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'success': False, 'error': 'Failed to read frame'}), 400

        # OpenCV BGR -> RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 추론 실행 (pipe_video_inspector.py와 동일: RGB NumPy 배열 직접 사용)
        print(f"[AI] Processing image... Input shape: {frame_rgb.shape}")
        inputs = segformer_processor(images=frame_rgb, return_tensors="pt")
        inputs = {k: v.to(segformer_device) for k, v in inputs.items()}
        print(f"[AI] Processor output shape: {inputs['pixel_values'].shape}")

        with torch.no_grad():
            outputs = segformer_model(**inputs)
            logits = outputs.logits

        # 결과를 원본 크기로 리사이즈 (pipe_video_inspector.py와 동일: frame.shape[:2])
        print(f"[AI] Logits shape before upsample: {logits.shape}")
        print(f"[AI] Target size (frame.shape[:2]): {frame.shape[:2]}")
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=frame.shape[:2],  # (height, width)
            mode="bilinear",
            align_corners=False
        )
        print(f"[AI] Upsampled logits shape: {upsampled_logits.shape}")
        print(f"[AI] Predicted mask shape: {upsampled_logits.shape[2:]}")

        # Softmax를 적용하여 확률로 변환 (pipe_video_inspector.py와 동일)
        probs = torch.nn.functional.softmax(upsampled_logits, dim=1)

        # 세그멘테이션 마스크 생성
        predicted = probs.argmax(dim=1)[0].cpu().numpy()

        # 컬러맵 적용 (파이프 세그멘테이션용)
        colored_mask = np.zeros((predicted.shape[0], predicted.shape[1], 3), dtype=np.uint8)

        # 커스텀 파이프 모델용 색상 (3 클래스: 배경, rust, scale)
        num_classes = predicted.max() + 1
        colors = np.array([
            [0, 0, 0],       # 클래스 0: background (검은색 - 표시 안됨)
            [255, 0, 0],     # 클래스 1: rust (빨간색)
            [255, 255, 0]    # 클래스 2: scale (노란색)
        ], dtype=np.uint8)

        for class_id in range(min(num_classes, len(colors))):
            mask = predicted == class_id
            colored_mask[mask] = colors[class_id]

        # 원본 이미지와 오버레이 (배경 제외 모든 영역)
        alpha = 0.5

        # 전체 이미지에 대해 블렌딩 수행
        blended = cv2.addWeighted(frame_rgb, 1 - alpha, colored_mask, alpha, 0)

        # 배경이 아닌 영역만 선택적으로 오버레이
        overlay = frame_rgb.copy()
        non_background_mask = predicted != 0

        # 마스크가 있는 경우만 오버레이 적용
        if np.any(non_background_mask):
            overlay[non_background_mask] = blended[non_background_mask]

        # JPEG로 인코딩
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Base64 인코딩
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 마스크를 PNG로 인코딩 (그레이스케일, 픽셀값 = 클래스 ID)
        mask_png = Image.fromarray(predicted.astype(np.uint8), mode='L')
        mask_buffer = BytesIO()
        mask_png.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')

        # 바운딩 박스 및 마스크 추출 (마스크 포함)
        bounding_boxes = extract_bounding_boxes_from_mask(predicted, min_area=100, include_masks=True)

        print(f"[AI] Inference completed. Found {num_classes} classes and {len(bounding_boxes)} objects")

        return jsonify({
            'success': True,
            'image': img_base64,
            'mask': mask_base64,
            'num_classes': int(num_classes),
            'width': int(predicted.shape[1]),
            'height': int(predicted.shape[0]),
            'bounding_boxes': bounding_boxes
        })

    except Exception as e:
        print(f"[AI] Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ai/inference_box', methods=['POST'])
def run_inference_on_box():
    """바운딩 박스 영역에 대해서만 SegFormer 추론 실행"""
    global inference_stats

    # 추론 락 획득 (순차 처리)
    with inference_lock:
        inference_stats['total_requests'] += 1
        inference_stats['active_requests'] += 1
        inference_stats['max_concurrent'] = max(inference_stats['max_concurrent'], inference_stats['active_requests'])

        try:
            return _run_inference_on_box_internal()
        finally:
            inference_stats['active_requests'] -= 1

def _run_inference_on_box_internal():
    """내부 박스 추론 로직 (락으로 보호됨)"""
    global segformer_model, segformer_processor, segformer_device, ai_initialized

    try:
        if not ai_initialized or segformer_model is None:
            return jsonify({
                'success': False,
                'error': 'AI model not initialized. Call /api/ai/initialize first.'
            }), 400

        data = request.json
        project_dir = data.get('project_dir')
        video_id = data.get('video_id')
        frame_number = data.get('frame_number')
        box = data.get('box')  # {x, y, width, height}

        if not all([project_dir, video_id, frame_number is not None, box]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: project_dir, video_id, frame_number, box'
            }), 400

        print(f"[AI] Running inference on box region at frame {frame_number}: ({box['x']}, {box['y']}, {box['width']}, {box['height']})")

        # project.json에서 직접 프로젝트 정보 읽기
        from pathlib import Path
        import json

        project_json_path = Path(project_dir) / 'project.json'
        if not project_json_path.exists():
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        with open(project_json_path, 'r', encoding='utf-8') as f:
            project_data = json.load(f)

        # 비디오 정보 찾기
        video_info = None
        for video in project_data.get('videos', []):
            if video.get('video_id') == video_id:
                video_info = video
                break

        if not video_info:
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        video_path = video_info.get('video_path')
        if not video_path or not os.path.exists(video_path):
            return jsonify({'success': False, 'error': f'Video file not found: {video_path}'}), 404

        # 프레임 추출
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Failed to open video'}), 500

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return jsonify({'success': False, 'error': 'Failed to read frame'}), 400

        # 박스 영역 크롭
        x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])

        # 경계 체크
        frame_h, frame_w = frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        cropped_frame = frame[y:y+h, x:x+w]

        # OpenCV BGR -> RGB 변환
        cropped_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        # 추론 실행 (pipe_video_inspector.py와 동일: RGB NumPy 배열 직접 사용)
        print("[AI] Processing cropped region...")
        inputs = segformer_processor(images=cropped_rgb, return_tensors="pt")
        inputs = {k: v.to(segformer_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = segformer_model(**inputs)
            logits = outputs.logits

        # 결과를 크롭 영역 크기로 리사이즈 (pipe_video_inspector.py와 동일: frame.shape[:2])
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=cropped_frame.shape[:2],  # (height, width)
            mode="bilinear",
            align_corners=False
        )

        # Softmax를 적용하여 확률로 변환 (pipe_video_inspector.py와 동일)
        probs = torch.nn.functional.softmax(upsampled_logits, dim=1)

        # 세그멘테이션 마스크 생성
        predicted = probs.argmax(dim=1)[0].cpu().numpy()

        # 전체 프레임 크기의 마스크 생성 (배경으로 초기화)
        full_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
        full_mask[y:y+h, x:x+w] = predicted

        # 박스 영역 내에서 가장 많이 감지된 클래스 찾기
        unique, counts = np.unique(predicted, return_counts=True)
        class_counts = dict(zip(unique, counts))

        # 클래스 이름 매핑
        class_names = {
            0: 'background',
            1: 'rust',
            2: 'scale'
        }

        # 배경을 제외한 클래스 중에서 가장 많이 감지된 클래스 찾기
        non_bg_counts = {k: v for k, v in class_counts.items() if k != 0}

        if non_bg_counts:
            # 배경이 아닌 클래스가 있으면 그중 가장 많은 것 선택
            dominant_class = max(non_bg_counts, key=non_bg_counts.get)
            dominant_class_ratio = non_bg_counts[dominant_class] / predicted.size
            dominant_class_name = class_names.get(dominant_class, f'class_{dominant_class}')
        else:
            # 배경만 있는 경우 - 이 경우는 어노테이션을 만들지 않음
            dominant_class = 0
            dominant_class_ratio = 1.0
            dominant_class_name = 'background'

        # 컬러 마스크 생성
        colored_mask = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        colors = np.array([
            [0, 0, 0],       # 클래스 0: background (검은색 - 표시 안됨)
            [255, 0, 0],     # 클래스 1: rust (빨간색)
            [255, 255, 0]    # 클래스 2: scale (노란색)
        ], dtype=np.uint8)

        num_classes = predicted.max() + 1
        for class_id in range(min(num_classes, len(colors))):
            mask = full_mask == class_id
            colored_mask[mask] = colors[class_id]

        # 원본 프레임 RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 박스 영역에만 오버레이 적용
        alpha = 0.5
        overlay = frame_rgb.copy()

        # 배경이 아닌 영역만 오버레이
        non_background_mask = full_mask != 0
        if np.any(non_background_mask):
            blended = cv2.addWeighted(frame_rgb, 1 - alpha, colored_mask, alpha, 0)
            overlay[non_background_mask] = blended[non_background_mask]

        # JPEG로 인코딩
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', overlay_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Base64 인코딩
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # 마스크도 Base64로 인코딩 (PNG로 저장 - 무손실)
        _, mask_buffer = cv2.imencode('.png', full_mask)
        mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')

        # 폴리곤 추출 (배경이 아닌 경우에만)
        polygon_points = []
        if dominant_class != 0:  # background가 아닌 경우
            # 지배적인 클래스의 마스크 영역에서 윤곽선 추출
            class_mask = (full_mask == dominant_class).astype(np.uint8) * 255
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 가장 큰 윤곽선 선택
                largest_contour = max(contours, key=cv2.contourArea)

                # 폴리곤 단순화
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                # 폴리곤 포인트를 리스트로 변환
                for point in approx_contour:
                    polygon_points.append({
                        'x': int(point[0][0]),
                        'y': int(point[0][1])
                    })

                print(f"[POLYGON] Extracted {len(polygon_points)} points for {dominant_class_name}")

        print(f"[AI] Box region inference completed. Found {num_classes} classes, dominant: {dominant_class_name} ({dominant_class_ratio*100:.1f}%)")

        return jsonify({
            'success': True,
            'image': img_base64,
            'mask': mask_base64,
            'polygon': polygon_points,
            'num_classes': int(num_classes),
            'width': int(frame_w),
            'height': int(frame_h),
            'dominant_class': int(dominant_class),
            'dominant_class_name': dominant_class_name,
            'dominant_class_ratio': float(dominant_class_ratio)
        })

    except Exception as e:
        print(f"[AI] Error during box inference: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/polygon/generate_mask', methods=['POST'])
def generate_mask_from_polygon():
    """폴리곤에서 마스크 생성"""
    try:
        data = request.json
        polygon = data.get('polygon', [])
        width = data.get('width')
        height = data.get('height')
        class_id = data.get('class_id', 1)

        if not polygon or not width or not height:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters: polygon, width, height'
            }), 400

        print(f"[MASK] Generating mask from polygon with {len(polygon)} points ({width}x{height})")

        # 폴리곤 포인트를 NumPy 배열로 변환
        import cv2
        points = np.array([[int(p['x']), int(p['y'])] for p in polygon], dtype=np.int32)

        # 마스크 생성 (빈 이미지)
        mask = np.zeros((height, width), dtype=np.uint8)

        # 폴리곤 채우기
        cv2.fillPoly(mask, [points], 255)

        # PNG로 인코딩
        mask_png = Image.fromarray(mask, mode='L')
        mask_buffer = BytesIO()
        mask_png.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')

        print(f"[MASK] Mask generated successfully")

        return jsonify({
            'success': True,
            'mask': mask_base64,
            'width': width,
            'height': height
        })

    except Exception as e:
        print(f"[MASK] Error generating mask from polygon: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export/dataset', methods=['POST'])
def export_dataset():
    """어노테이션 데이터를 학습 데이터셋으로 export (SegFormer, YOLO)"""
    try:
        data = request.json
        project_id = data.get('project_id')
        video_id = data.get('video_id')
        annotations = data.get('annotations', {})
        export_format = data.get('format', 'segformer')  # 'segformer' or 'yolo'
        output_dir = data.get('output_dir', 'datasets')

        if not all([project_id, video_id, annotations]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400

        print(f"[EXPORT] Exporting dataset for {project_id}/{video_id} in {export_format} format")

        # 프로젝트 매니저로 비디오 정보 가져오기
        pm = ProjectManager()
        projects = pm.list_projects()

        project = None
        for p in projects:
            if p.id == project_id:
                project = p
                break

        if not project:
            return jsonify({'success': False, 'error': 'Project not found'}), 404

        video_info = None
        for video in project.videos:
            if video.get('video_id') == video_id:
                video_info = video
                break

        if not video_info:
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        from pathlib import Path
        video_path = str((Path(project.project_dir) / 'videos' / video_info['filename']).resolve())

        if not os.path.exists(video_path):
            return jsonify({'success': False, 'error': f'Video file not found: {video_path}'}), 404

        # 출력 디렉토리 생성
        dataset_name = f"{project_id}_{video_id}_{export_format}"
        output_path = Path(output_dir) / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)

        images_dir = output_path / 'images'
        images_dir.mkdir(exist_ok=True)

        if export_format == 'segformer':
            masks_dir = output_path / 'masks'
            masks_dir.mkdir(exist_ok=True)
        elif export_format == 'yolo':
            labels_dir = output_path / 'labels'
            labels_dir.mkdir(exist_ok=True)

        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Failed to open video'}), 500

        exported_count = 0
        frame_metadata = []

        # 어노테이션이 있는 프레임만 처리
        for frame_num_str, frame_annotations in annotations.items():
            frame_num = int(frame_num_str)

            # 사용자가 추가한 어노테이션만 export (auto_detected 제외)
            user_annotations = [anno for anno in frame_annotations if not anno.get('auto_detected', False)]

            if not user_annotations:
                continue

            # 프레임 추출
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                print(f"[EXPORT] Failed to read frame {frame_num}")
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_height, frame_width = frame.shape[:2]

            # 이미지 저장
            image_filename = f"frame_{frame_num:06d}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), frame)

            if export_format == 'segformer':
                # SegFormer 형식: 픽셀 단위 세그멘테이션 마스크 생성
                mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

                for anno in user_annotations:
                    if anno.get('has_segmentation') and anno.get('mask'):
                        # 마스크 디코딩
                        mask_data = base64.b64decode(anno['mask'])
                        mask_img = Image.open(BytesIO(mask_data))
                        mask_array = np.array(mask_img)

                        # 바운딩 박스 위치
                        box = anno['box']
                        x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])

                        # 마스크 리사이즈
                        if mask_array.shape != (h, w):
                            mask_img_resized = mask_img.resize((w, h), Image.NEAREST)
                            mask_array = np.array(mask_img_resized)

                        # 클래스 ID 적용 (마스크가 0이 아닌 곳에만)
                        class_id = anno.get('class_id', 1)
                        mask_binary = mask_array > 0
                        mask[y:y+h, x:x+w][mask_binary] = class_id

                # 마스크 저장
                mask_filename = f"frame_{frame_num:06d}.png"
                mask_path = masks_dir / mask_filename
                Image.fromarray(mask, mode='L').save(mask_path)

                frame_metadata.append({
                    'frame': frame_num,
                    'image': image_filename,
                    'mask': mask_filename,
                    'annotations': len(user_annotations)
                })

            elif export_format == 'yolo':
                # YOLO Segmentation 형식: normalized polygon coordinates
                label_filename = f"frame_{frame_num:06d}.txt"
                label_path = labels_dir / label_filename

                with open(label_path, 'w') as f:
                    for anno in user_annotations:
                        if anno.get('has_segmentation') and anno.get('mask'):
                            class_id = anno.get('class_id', 1)

                            # 마스크 디코딩
                            mask_data = base64.b64decode(anno['mask'])
                            mask_img = Image.open(BytesIO(mask_data))
                            mask_array = np.array(mask_img)

                            # 바운딩 박스 위치
                            box = anno['box']
                            x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])

                            # 마스크 리사이즈
                            if mask_array.shape != (h, w):
                                mask_img_resized = mask_img.resize((w, h), Image.NEAREST)
                                mask_array = np.array(mask_img_resized)

                            # 컨투어 추출
                            mask_binary = (mask_array > 0).astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            if contours:
                                # 가장 큰 컨투어 선택
                                contour = max(contours, key=cv2.contourArea)

                                # Polygon points (normalized)
                                points = []
                                for point in contour.squeeze():
                                    if len(point.shape) == 1 and len(point) == 2:
                                        # 절대 좌표
                                        abs_x = x + point[0]
                                        abs_y = y + point[1]
                                        # Normalize
                                        norm_x = abs_x / frame_width
                                        norm_y = abs_y / frame_height
                                        points.extend([norm_x, norm_y])

                                if points:
                                    # YOLO format: class_id x1 y1 x2 y2 ... xn yn
                                    line = f"{class_id} " + " ".join(f"{p:.6f}" for p in points)
                                    f.write(line + "\n")

                frame_metadata.append({
                    'frame': frame_num,
                    'image': image_filename,
                    'label': label_filename,
                    'annotations': len(user_annotations)
                })

            exported_count += 1

        cap.release()

        # 메타데이터 저장
        import json
        from datetime import datetime
        metadata = {
            'project_id': project_id,
            'video_id': video_id,
            'format': export_format,
            'total_frames': exported_count,
            'class_names': {
                0: 'background',
                1: 'rust',
                2: 'scale'
            },
            'frames': frame_metadata,
            'exported_at': datetime.now().isoformat()
        }

        metadata_path = output_path / 'dataset_info.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"[EXPORT] Dataset exported: {exported_count} frames to {output_path}")

        return jsonify({
            'success': True,
            'format': export_format,
            'output_dir': str(output_path),
            'frames_exported': exported_count,
            'metadata_file': str(metadata_path)
        })

    except Exception as e:
        print(f"[EXPORT] Error exporting dataset: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def extract_frame_with_ffmpeg(video_path: str, frame_number: int) -> bytes:
    """FFmpeg을 사용해 특정 프레임을 JPEG로 추출 (정확도 우선)"""
    filter_chain = f"select='eq(n\\,{frame_number})',scale=640:-1"
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', video_path,
        '-vf', filter_chain,
        '-vsync', '0',
        '-frames:v', '1',
        '-f', 'image2pipe',
        '-vcodec', 'mjpeg',
        'pipe:1'
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=15
    )

    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(stderr or 'ffmpeg failed')

    return result.stdout


@app.route('/api/inference/status/<job_id>', methods=['GET'])
def get_inference_status(job_id):
    """추론 작업 상태 조회"""
    with job_lock:
        if job_id not in active_jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

        job = active_jobs[job_id]
        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': job['status'],
            'progress': job['progress'],
            'current_frame': job.get('current_frame', 0),
            'total_frames': job.get('total_frames', 0),
            'video_path': job.get('video_path', ''),
            'output_path': job.get('output_path', '')
        })


@app.route('/api/inference/cancel/<job_id>', methods=['POST'])
def cancel_inference(job_id):
    """추론 작업 취소"""
    with job_lock:
        if job_id not in active_jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

        job = active_jobs[job_id]
        if job['status'] == 'completed':
            return jsonify({
                'success': False,
                'error': 'Job already completed'
            }), 400

        job['cancel_requested'] = True
        job['status'] = 'cancelling'


        return jsonify({
            'success': True,
            'message': 'Cancel request sent',
            'job_id': job_id
        })


@app.route('/api/inference/preview/<job_id>', methods=['GET'])
def get_inference_preview(job_id):
    """추론 작업의 최신 처리된 프레임 이미지 반환"""
    with job_lock:
        if job_id not in active_jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

        job = active_jobs[job_id]
        latest_frame_path = job.get('latest_frame_path')

        if not latest_frame_path or not os.path.exists(latest_frame_path):
            return jsonify({
                'success': False,
                'error': 'No preview frame available yet'
            }), 404

    # 이미지 파일 반환
    try:
        return send_file(latest_frame_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/frames/<job_id>', methods=['GET'])
def get_inference_frames(job_id):
    """추론 작업의 처리된 프레임 정보 반환 (비디오 플레이어용)"""
    with job_lock:
        if job_id not in active_jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

        job = active_jobs[job_id]
        frame_paths = job.get('frame_paths', [])

        return jsonify({
            'success': True,
            'job_id': job_id,
            'status': job['status'],
            'total_frames': job.get('total_frames', 0),
            'processed_frames': len(frame_paths),
            'fps': job.get('fps', 30),
            'current_frame': job.get('current_frame', 0)
        })


@app.route('/api/inference/frame/<job_id>/<int:frame_index>', methods=['GET'])
def get_inference_frame(job_id, frame_index):
    """특정 프레임 이미지 반환 (비디오 플레이어용)"""
    with job_lock:
        if job_id not in active_jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

        job = active_jobs[job_id]
        frame_paths = job.get('frame_paths', [])

        if frame_index < 0 or frame_index >= len(frame_paths):
            return jsonify({
                'success': False,
                'error': f'Frame index {frame_index} out of range (0-{len(frame_paths)-1})'
            }), 404

        frame_path = frame_paths[frame_index]

    # 이미지 파일 반환
    try:
        if not os.path.exists(frame_path):
            return jsonify({
                'success': False,
                'error': 'Frame file not found'
            }), 404

        return send_file(frame_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/check', methods=['POST'])
def check_inference_results():
    """추론 결과가 이미 존재하는지 확인"""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        output_path = data.get('output_path')

        if not video_path or not output_path:
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400

        # 상대 경로를 절대 경로로 변환
        if not os.path.isabs(output_path):
            # 프로젝트 루트 디렉토리 기준으로 절대 경로 생성
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(base_dir, output_path)


        # 출력 디렉토리와 결과 파일 확인
        result_json_path = os.path.join(output_path, 'inference_results.json')

        if os.path.exists(result_json_path):
            # 결과 파일이 존재하면 메타데이터 읽기
            try:
                with open(result_json_path, 'r') as f:
                    result_data = json.load(f)

                # 프레임 파일들이 실제로 존재하는지 확인
                frame_count = 0
                frame_files = sorted([f for f in os.listdir(output_path) if f.startswith('frame_') and f.endswith('.jpg')])
                frame_count = len(frame_files)

                return jsonify({
                    'success': True,
                    'exists': True,
                    'result_path': output_path,
                    'total_frames': result_data.get('total_frames', frame_count),
                    'fps': result_data.get('fps', 30),
                    'video_path': result_data.get('video_path', video_path),
                    'frame_count': frame_count
                })
            except Exception as e:
                return jsonify({
                    'success': True,
                    'exists': False,
                    'error': 'Result file corrupted'
                })
        else:
            return jsonify({
                'success': True,
                'exists': False
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/completed-frame', methods=['POST'])
def get_completed_frame():
    """완료된 추론 결과 프레임 이미지 반환 (파일 경로 기반)"""
    try:
        data = request.get_json()
        frame_path = data.get('frame_path')

        if not frame_path:
            return jsonify({'success': False, 'error': 'Frame path required'}), 400

        # 상대 경로를 절대 경로로 변환
        if not os.path.isabs(frame_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            frame_path = os.path.join(base_dir, frame_path)


        if not os.path.exists(frame_path):
            return jsonify({'success': False, 'error': 'Frame file not found'}), 404

        return send_file(frame_path, mimetype='image/jpeg')

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def process_video_inference(job_id, video_path, output_path, model_type):
    """백그라운드에서 비디오 추론 실행"""
    global segformer_model, segformer_processor, segformer_device

    try:
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with job_lock:
                active_jobs[job_id]['status'] = 'failed'
                active_jobs[job_id]['error'] = 'Failed to open video file'
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 작업 정보 업데이트
        with job_lock:
            active_jobs[job_id]['total_frames'] = total_frames
            active_jobs[job_id]['fps'] = fps
            active_jobs[job_id]['frame_paths'] = []  # 모든 프레임 경로 저장


        # 결과 저장용
        results = []
        frame_count = 0

        # 프레임별 추론
        while True:
            # 취소 요청 확인
            with job_lock:
                if active_jobs[job_id]['cancel_requested']:
                    active_jobs[job_id]['status'] = 'cancelled'
                    cap.release()
                    return

            ret, frame = cap.read()
            if not ret:
                break

            # OpenCV BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 추론 실행
            inputs = segformer_processor(images=frame_rgb, return_tensors="pt")
            inputs = {k: v.to(segformer_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = segformer_model(**inputs)
                logits = outputs.logits

            # 결과를 원본 크기로 리사이즈
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=frame.shape[:2],  # (height, width)
                mode="bilinear",
                align_corners=False
            )

            # 예측 클래스 맵
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            # 클래스별 마스크 저장
            unique_classes = np.unique(pred_seg)
            frame_results = {
                'frame_number': frame_count,
                'classes': unique_classes.tolist()
            }

            # 마스크 이미지 저장 (시각화용)
            mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
            # rust: 빨강, scale: 노랑
            mask_colored[pred_seg == 1] = [255, 0, 0]  # rust
            mask_colored[pred_seg == 2] = [255, 255, 0]  # scale

            # 원본 프레임과 마스크 오버레이
            overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

            # 결과 이미지 저장
            output_frame_path = os.path.join(output_path, f'frame_{frame_count:06d}.jpg')
            cv2.imwrite(output_frame_path, overlay)

            results.append(frame_results)
            frame_count += 1

            # 진행 상황 업데이트
            progress = (frame_count / total_frames) * 100
            with job_lock:
                active_jobs[job_id]['current_frame'] = frame_count
                active_jobs[job_id]['progress'] = progress
                # 최신 처리된 프레임 경로 저장 (미리보기용)
                active_jobs[job_id]['latest_frame_path'] = output_frame_path
                # 모든 프레임 경로 저장 (비디오 플레이어용)
                active_jobs[job_id]['frame_paths'].append(output_frame_path)


        cap.release()

        # 결과 JSON 저장
        result_json_path = os.path.join(output_path, 'inference_results.json')
        with open(result_json_path, 'w') as f:
            json.dump({
                'video_path': video_path,
                'total_frames': frame_count,
                'fps': fps,
                'width': width,
                'height': height,
                'model_type': model_type,
                'results': results
            }, f, indent=2)

        # 작업 완료 상태 업데이트
        with job_lock:
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['progress'] = 100
            active_jobs[job_id]['result_file'] = result_json_path

    except Exception as e:
        with job_lock:
            if job_id in active_jobs:
                active_jobs[job_id]['status'] = 'failed'
                active_jobs[job_id]['error'] = str(e)


@app.route('/api/inference', methods=['POST'])
def run_video_inference():
    """전체 비디오에 대한 추론 실행 (비동기)"""
    global segformer_model, segformer_processor, segformer_device, ai_initialized

    try:
        if not ai_initialized or segformer_model is None:
            return jsonify({
                'success': False,
                'error': 'AI model not initialized. Call /api/ai/initialize first.'
            }), 400

        data = request.json
        print(f"[DEBUG] Inference request data: {data}", flush=True)

        model_type = data.get('model_type', 'segformer')
        model_path = data.get('model_path')
        video_path = data.get('video_path')
        output_path = data.get('output_path', 'inference_results')

        print(f"[DEBUG] Parsed - video_path: {video_path}, model_type: {model_type}", flush=True)

        if not video_path:
            return jsonify({
                'success': False,
                'error': 'Missing required parameter: video_path'
            }), 400

        # 상대 경로를 절대 경로로 변환
        if not os.path.isabs(video_path):
            # 부모 디렉토리 (pipe-inspector-electron) 기준으로 경로 변환
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            video_path = os.path.join(base_dir, video_path)

        print(f"[DEBUG] Resolved video_path: {video_path}", flush=True)
        print(f"[DEBUG] File exists: {os.path.exists(video_path)}", flush=True)

        # 출력 경로도 절대 경로로 변환
        if not os.path.isabs(output_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_path = os.path.join(base_dir, output_path)

        # 비디오 파일 존재 확인
        if not os.path.exists(video_path):
            return jsonify({
                'success': False,
                'error': f'Video file not found: {video_path}'
            }), 404

        # 작업 ID 생성
        import time
        import hashlib
        job_id = hashlib.md5(f"{video_path}_{time.time()}".encode()).hexdigest()[:16]

        # 작업 등록
        with job_lock:
            active_jobs[job_id] = {
                'status': 'running',
                'progress': 0,
                'current_frame': 0,
                'total_frames': 0,
                'video_path': video_path,
                'output_path': output_path,
                'cancel_requested': False
            }


        # 출력 디렉토리 생성
        os.makedirs(output_path, exist_ok=True)

        # 백그라운드 스레드로 추론 시작
        inference_thread = threading.Thread(
            target=process_video_inference,
            args=(job_id, video_path, output_path, model_type)
        )
        inference_thread.daemon = True
        inference_thread.start()

        # 즉시 job_id 반환
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Inference started',
            'video_path': video_path,
            'output_path': output_path
        })

    except Exception as e:
        # 에러 발생 시 작업 상태 업데이트
        try:
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id]['status'] = 'failed'
                    active_jobs[job_id]['error'] = str(e)
        except:
            pass

        return jsonify({
            'success': False,
            'error': str(e),
            'job_id': job_id if 'job_id' in locals() else None
        }), 500




@app.route('/api/dataset/build_yolo', methods=['POST'])
def build_yolo_dataset():
    """다중 프로젝트 YOLO 데이터셋 빌드"""
    from pathlib import Path
    import random
    import shutil
    from datetime import datetime

    try:
        data = request.get_json()
        annotations_data = data.get('annotations_data', [])
        output_dir = data.get('output_dir', 'pipe_dataset')
        split_ratio = data.get('split_ratio', '0.7,0.15,0.15')
        augment_multiplier = data.get('augment_multiplier', 0)
        base_projects_dir = Path(data.get('base_projects_dir', '/home/intu/Nas2/k_water/pipe_inspector_data'))

        if not annotations_data:
            return jsonify({'success': False, 'error': 'No annotations data provided'}), 400

        print(f"[DATASET BUILD] Building YOLO dataset from {len(annotations_data)} annotation files")

        # 출력 디렉토리 설정
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_dir

        # 기존 디렉토리가 있으면 타임스탬프 추가
        if output_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_path.parent / f"{output_path.name}_{timestamp}"

        # 디렉토리 구조 생성
        (output_path / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (output_path / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        (output_path / 'test' / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / 'test' / 'labels').mkdir(parents=True, exist_ok=True)

        print(f"[DATASET BUILD] Output directory: {output_path}")

        # Split ratio 파싱
        try:
            train_ratio, val_ratio, test_ratio = map(float, split_ratio.split(','))
            total_ratio = train_ratio + val_ratio + test_ratio
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        except:
            train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

        print(f"[DATASET BUILD] Split ratio: Train={train_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")

        # 프로젝트별 클래스 정의 수집
        project_classes = {}  # project_dir -> class_id_to_name mapping

        # 실제 사용된 클래스 수집
        used_classes = set()

        # 모든 어노테이션 프레임 수집
        all_frames = []
        for anno_data in annotations_data:
            user_id = anno_data['user_id']
            project_id = anno_data['project_id']
            video_id = anno_data['video_id']
            annotations = anno_data['annotations']
            project_dir = Path(anno_data['project_dir'])

            # 비디오 정보 및 클래스 정의 찾기
            project_file = project_dir / 'project.json'
            video_path = None

            if project_file.exists():
                with open(project_file, 'r', encoding='utf-8') as f:
                    project_json = json.load(f)

                    # 프로젝트의 클래스 정의 읽기 (처음 한 번만)
                    project_dir_str = str(project_dir)
                    if project_dir_str not in project_classes:
                        classes = project_json.get('classes', [])
                        class_mapping = {}
                        for idx, cls in enumerate(classes):
                            class_name = cls.get('name', f'class_{idx}')
                            class_mapping[idx] = class_name
                        project_classes[project_dir_str] = class_mapping
                        print(f"[DATASET BUILD] Loaded {len(class_mapping)} classes from project {project_id}")

                    # 비디오 경로 찾기
                    for video in project_json.get('videos', []):
                        if video.get('video_id') == video_id:
                            video_path = video.get('video_path')
                            break

            if not video_path:
                print(f"[DATASET BUILD] Warning: Video path not found for {video_id}")
                continue

            # 웹 호환 비디오 경로로 변환
            video_path_obj = Path(video_path)
            if 'SAHARA' in str(video_path):
                parts_list = list(video_path_obj.parts)
                sahara_idx = parts_list.index('SAHARA')
                relative_path = Path(*parts_list[sahara_idx+1:])
                web_video_path = Path('/home/intu/nas2_kwater/Videos_web/SAHARA') / relative_path
                web_video_path = web_video_path.with_suffix('.mp4')
            elif '관내시경영상' in str(video_path):
                parts_list = list(video_path_obj.parts)
                kwan_idx = parts_list.index('관내시경영상')
                relative_path = Path(*parts_list[kwan_idx+1:])
                web_video_path = Path('/home/intu/nas2_kwater/Videos_web/관내시경영상') / relative_path
                web_video_path = web_video_path.with_suffix('.mp4')
            else:
                web_video_path = Path(str(video_path).replace('.avi', '.mp4').replace('.AVI', '.mp4'))

            # 각 프레임에 대해
            for frame_num_str, frame_annos in annotations.items():
                if not frame_annos:
                    continue

                frame_num = int(frame_num_str)

                # 사용된 클래스 수집
                for anno in frame_annos:
                    class_id = anno.get('class_id', 0)
                    used_classes.add(class_id)

                all_frames.append({
                    'user_id': user_id,
                    'project_id': project_id,
                    'video_id': video_id,
                    'video_path': str(web_video_path),
                    'frame_num': frame_num,
                    'annotations': frame_annos,
                    'project_dir': str(project_dir)
                })

        if not all_frames:
            return jsonify({'success': False, 'error': 'No frames with annotations found'}), 400

        print(f"[DATASET BUILD] Total frames: {len(all_frames)}")

        # 프레임을 무작위로 섞기
        random.shuffle(all_frames)

        # Train/Val/Test 분할
        train_end = int(len(all_frames) * train_ratio)
        val_end = train_end + int(len(all_frames) * val_ratio)

        train_frames = all_frames[:train_end]
        val_frames = all_frames[train_end:val_end]
        test_frames = all_frames[val_end:]

        print(f"[DATASET BUILD] Train: {len(train_frames)}, Val: {len(val_frames)}, Test: {len(test_frames)}")

        # 각 세트별로 이미지 및 라벨 저장
        def process_frames(frames, split_name):
            saved_count = 0
            for idx, frame_data in enumerate(frames):
                try:
                    video_path = frame_data['video_path']
                    frame_num = frame_data['frame_num']
                    annotations = frame_data['annotations']

                    # 비디오에서 프레임 추출
                    cap = cv2.VideoCapture(video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    cap.release()

                    if not ret:
                        print(f"[DATASET BUILD] Failed to extract frame {frame_num} from {video_path}")
                        continue

                    # 이미지 파일명
                    image_filename = f"{frame_data['project_id']}_{frame_data['video_id']}_frame{frame_num}.jpg"
                    image_path = output_path / split_name / 'images' / image_filename
                    label_path = output_path / split_name / 'labels' / image_filename.replace('.jpg', '.txt')

                    # 이미지 저장
                    cv2.imwrite(str(image_path), frame)

                    # YOLO 라벨 생성
                    height, width = frame.shape[:2]
                    yolo_labels = []

                    for anno in annotations:
                        if not anno.get('polygon'):
                            continue

                        # 클래스 ID (label 필드에서 추출, 없으면 0)
                        class_id = anno.get('class_id', 0)

                        # 폴리곤 좌표 정규화
                        polygon = anno['polygon']
                        normalized_coords = []
                        for point in polygon:
                            x_norm = point['x'] / width
                            y_norm = point['y'] / height
                            normalized_coords.append(f"{x_norm:.6f} {y_norm:.6f}")

                        # YOLO segmentation 형식: class_id x1 y1 x2 y2 ... xn yn
                        yolo_line = f"{class_id} " + " ".join(normalized_coords)
                        yolo_labels.append(yolo_line)

                    # 라벨 파일 저장
                    if yolo_labels:
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(yolo_labels))
                        saved_count += 1

                except Exception as e:
                    print(f"[DATASET BUILD] Error processing frame: {e}")
                    continue

            return saved_count

        # 각 세트 처리
        train_count = process_frames(train_frames, 'train')
        val_count = process_frames(val_frames, 'val')
        test_count = process_frames(test_frames, 'test')

        print(f"[DATASET BUILD] Saved - Train: {train_count}, Val: {val_count}, Test: {test_count}")

        # 모든 프로젝트의 클래스 매핑을 병합
        merged_class_mapping = {}
        for project_dir_str, class_mapping in project_classes.items():
            merged_class_mapping.update(class_mapping)

        # 사용된 클래스 정보 정리
        sorted_class_ids = sorted(used_classes)
        class_names_list = [merged_class_mapping.get(cid, f'class_{cid}') for cid in sorted_class_ids]
        num_classes = len(sorted_class_ids)

        print(f"[DATASET BUILD] Used classes ({num_classes}): {sorted_class_ids}")
        print(f"[DATASET BUILD] Class names: {class_names_list}")

        # data.yaml 생성
        yaml_content = f"""# YOLO Dataset Configuration
path: {output_path}
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {num_classes}

# Class names
names: {class_names_list}
"""

        with open(output_path / 'data.yaml', 'w') as f:
            f.write(yaml_content)

        # dataset_info.json 생성
        info = {
            'created_at': datetime.now().isoformat(),
            'total_frames': len(all_frames),
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'split_ratio': split_ratio,
            'format': 'yolo_segmentation',
            'augment_multiplier': augment_multiplier,
            'num_classes': num_classes,
            'class_names': class_names_list,
            'class_ids': sorted_class_ids
        }

        with open(output_path / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"[DATASET BUILD] ✅ Dataset build complete: {output_path}")

        return jsonify({
            'success': True,
            'output_dir': str(output_path),
            'total_images': train_count + val_count + test_count,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count
        })

    except Exception as e:
        print(f"[DATASET BUILD] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting GPU Server API...")
    print("📡 API Server: http://0.0.0.0:5004")
    print("🎮 GPU Tasks: Enabled")

    # AI 모델 자동 초기화
    print("\n🤖 Loading AI model...")
    if load_ai_model():
        print("✅ AI model ready\n")
    else:
        print("⚠️  AI model failed to load (will retry on first inference)\n")

    # 멀티스레드 활성화로 동시 요청 처리 가능
    app.run(host='0.0.0.0', port=5004, debug=False, threaded=True)
