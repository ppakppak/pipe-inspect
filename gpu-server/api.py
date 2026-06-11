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
import time

# Grounded-SAM 경로 추가
# sys.path.insert(0, '/home/ppak/SynologyDrive/ykpark/linux_devel/ground_sam/Grounded-Segment-Anything')

from project_manager import ProjectManager
from defect_sizing import VanishingPointDetector, DepthEstimator, PipeSizeCalibrator, PipeAreaRatioCalculator, PipeUnwrapper, DepthAwarePipeUnwrapper, SizingResultManager

# SegFormer 모델 전역 변수
segformer_model = None
segformer_processor = None
segformer_device = None
ai_initialized = False

# YOLO 모델 전역 변수
yolo_model = None
yolo_initialized = False

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

# Sizing helpers
vp_detector = VanishingPointDetector()

app = Flask(__name__)
CORS(app)

# 대용량 JSON 요청 허용 (데이터셋 빌드 시 100MB까지)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB


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
            checkpoint = torch.load(model_path, map_location=segformer_device, weights_only=False)
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
                checkpoint = torch.load(model_path, map_location=segformer_device, weights_only=False)

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


def load_yolo_model(model_path=None):
    """YOLO 모델 로드"""
    global yolo_model, yolo_initialized, segformer_device

    try:
        from ultralytics import YOLO

        print("[AI] Initializing YOLO model...")

        # 디바이스 설정 (SegFormer와 공유)
        if segformer_device is None:
            if torch.cuda.is_available():
                segformer_device = torch.device("cuda")
                print(f"[AI] YOLO using GPU")
            else:
                segformer_device = torch.device("cpu")
                print("[AI] YOLO using CPU")

        # 모델 경로 결정
        if model_path and os.path.exists(model_path):
            print(f"[AI] Loading custom YOLO model from: {model_path}")
            yolo_model = YOLO(model_path)
        else:
            # 기본 경로에서 모델 찾기
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_paths = [
                os.path.join(script_dir, 'yolo_best.pt'),
                os.path.join(script_dir, 'best.pt'),
                os.path.join(script_dir, 'runs', 'segment', 'train', 'weights', 'best.pt'),
            ]

            model_found = False
            for path in default_paths:
                if os.path.exists(path):
                    print(f"[AI] Loading YOLO model from: {path}")
                    yolo_model = YOLO(path)
                    model_found = True
                    break

            if not model_found:
                # YOLOv8 사전학습 모델 사용 (세그멘테이션)
                print("[AI] No custom YOLO model found, using pretrained yolov8n-seg")
                yolo_model = YOLO('yolov8n-seg.pt')

        yolo_initialized = True
        print("[AI] YOLO model initialized successfully")
        return True

    except ImportError:
        print("[AI] Error: ultralytics package not installed. Run: pip install ultralytics")
        return False
    except Exception as e:
        print(f"[AI] Error initializing YOLO model: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/api/ai/initialize/yolo', methods=['POST'])
def initialize_yolo():
    """YOLO 모델 초기화 API 엔드포인트"""
    global yolo_initialized

    data = request.json or {}
    model_path = data.get('model_path')

    if yolo_initialized:
        return jsonify({
            'success': True,
            'message': 'YOLO model already initialized'
        })

    success = load_yolo_model(model_path)

    if success:
        return jsonify({
            'success': True,
            'message': 'YOLO model initialized'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'Failed to initialize YOLO model'
        }), 500


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





def _get_project_video_path(project_dir, video_id):
    from pathlib import Path
    import json

    project_json_path = Path(project_dir) / 'project.json'
    if not project_json_path.exists():
        raise FileNotFoundError(f'Project not found: {project_json_path}')

    with open(project_json_path, 'r', encoding='utf-8') as f:
        project_data = json.load(f)

    for video in project_data.get('videos', []):
        if video.get('video_id') == video_id:
            video_path = video.get('video_path', str(Path(project_dir) / 'videos' / video['filename']))
            return video_path, int(video.get('total_frames', 0) or 0)

    raise FileNotFoundError(f'Video not found: {video_id}')


def _load_video_frame_for_sizing(project_dir, video_id, frame_number):
    video_path, total_frames = _get_project_video_path(project_dir, video_id)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file not found: {video_path}')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Failed to open video: {video_path}')

    if total_frames <= 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames > 0:
        frame_number = max(0, min(int(frame_number), total_frames - 1))
    else:
        frame_number = max(0, int(frame_number))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if (not ret or frame is None) and frame_number < 200:
        for fallback_frame in [10, 30, 50, 100, 150, 200]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback_frame)
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_number = fallback_frame
                break

    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f'Failed to read frame {frame_number}')

    return frame, frame_number, total_frames, video_path


@app.route('/api/sizing/detect-vp', methods=['POST'])
def detect_vp_api():
    try:
        data = request.json or {}
        project_dir = data.get('project_dir')
        video_id = data.get('video_id')
        frame_number = int(data.get('frame_number', 0))

        if not project_dir:
            return jsonify({'success': False, 'error': 'project_dir required'}), 400
        if not video_id:
            return jsonify({'success': False, 'error': 'video_id required'}), 400

        frame, actual_frame, total_frames, video_path = _load_video_frame_for_sizing(project_dir, video_id, frame_number)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_masked = vp_detector._mask_osd(gray)

        result_radial = vp_detector._detect_radial_convergence(gray_masked)
        result_dark = vp_detector._detect_gaussian_darkest(gray_masked)
        result = result_radial if result_radial['confidence'] >= result_dark['confidence'] else result_dark

        cache_key = f"{project_dir}:{video_id}"
        vp_detector._cache[cache_key] = { 'vp': result, 'timestamp': time.time() }

        return jsonify({
            'success': True,
            'vp': result,
            'vp_radial': result_radial,
            'vp_darkest': result_dark,
            'frame_number': actual_frame,
            'requested_frame_number': frame_number,
            'total_frames': total_frames,
            'video_path': video_path,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sizing/detect-vp-batch', methods=['POST'])
def detect_vp_batch_api():
    try:
        data = request.json or {}
        project_dir = data.get('project_dir')
        video_id = data.get('video_id')
        frame_numbers = data.get('frame_numbers') or []

        if not project_dir:
            return jsonify({'success': False, 'error': 'project_dir required'}), 400
        if not video_id:
            return jsonify({'success': False, 'error': 'video_id required'}), 400
        if not frame_numbers:
            return jsonify({'success': False, 'error': 'frame_numbers required'}), 400

        frames = []
        used_frames = []
        per_frame_results = []

        for fn in frame_numbers:
            try:
                frame, actual_frame, _, _ = _load_video_frame_for_sizing(project_dir, video_id, int(fn))
                frames.append(frame)
                used_frames.append(actual_frame)
            except Exception:
                continue

        if not frames:
            return jsonify({'success': False, 'error': 'No valid frames loaded'}), 400

        batch_result = vp_detector.detect_batch(frames, video_id=f"{project_dir}:{video_id}")

        for actual_frame, frame in zip(used_frames, frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_masked = vp_detector._mask_osd(gray)
            result_radial = vp_detector._detect_radial_convergence(gray_masked)
            result_dark = vp_detector._detect_gaussian_darkest(gray_masked)
            chosen = result_radial if result_radial['confidence'] >= result_dark['confidence'] else result_dark
            per_frame_results.append({
                'frame_number': actual_frame,
                'vp': chosen,
                'vp_radial': result_radial,
                'vp_darkest': result_dark,
            })

        return jsonify({
            'success': True,
            'frames_processed': len(frames),
            'frame_numbers': used_frames,
            'results': per_frame_results,
            'median_vp': batch_result.get('median_vp'),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500




def _encode_image_base64(image, ext='.jpg', params=None):
    if params is None:
        params = [cv2.IMWRITE_JPEG_QUALITY, 90] if ext == '.jpg' else []
    ok, buf = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError('Failed to encode image')
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def _flat_polygon_to_points(polygon_flat):
    pts = []
    for i in range(0, len(polygon_flat), 2):
        pts.append([float(polygon_flat[i]), float(polygon_flat[i+1])])
    return pts


def _normalize_defect_polygon(defect):
    polygon = defect.get('polygon') or defect.get('segmentation')
    if isinstance(polygon, list) and polygon and isinstance(polygon[0], dict):
        flat = []
        for pt in polygon:
            flat.extend([float(pt['x']), float(pt['y'])])
        polygon = flat
    return polygon if isinstance(polygon, list) and len(polygon) >= 6 else None


def _resolve_vp_from_request(data, frame, video_cache_key):
    req_vp = data.get('vp') or {}
    if req_vp and 'vp_x' in req_vp and 'vp_y' in req_vp:
        return {
            'vp_x': float(req_vp['vp_x']),
            'vp_y': float(req_vp['vp_y']),
            'confidence': float(req_vp.get('confidence', 1.0)),
            'method': req_vp.get('method', 'client')
        }
    return vp_detector.detect(frame, video_id=video_cache_key)


def _measure_defects(defects, calibrator, depth_map=None, vp=None):
    measurements = []
    for defect in defects:
        polygon_flat = _normalize_defect_polygon(defect)
        if not polygon_flat:
            measurements.append({
                'annotation_index': defect.get('annotation_index', defect.get('index', -1)),
                'label': defect.get('label', defect.get('category', 'unknown')),
                'error': 'invalid polygon'
            })
            continue

        result = calibrator.measure_defect(_flat_polygon_to_points(polygon_flat), depth_map=depth_map)
        result['annotation_index'] = defect.get('annotation_index', defect.get('index', -1))
        result['label'] = defect.get('label', defect.get('category', 'unknown'))
        if vp:
            result['vp'] = vp
        measurements.append(result)
    return measurements


def _bbox_of_flat_polygon(poly_flat, mmpx_x=1.0, mmpx_y=1.0):
    """평면 좌표 폴리곤(flat) → 축정렬 bbox + 종횡비.

    Returns: {bbox_width_mm, bbox_height_mm, aspect_wh, polygon_fill_pct}
    """
    if not poly_flat or len(poly_flat) < 6:
        return {'bbox_width_mm': None, 'bbox_height_mm': None,
                'aspect_wh': None, 'polygon_fill_pct': None}
    xs = poly_flat[0::2]
    ys = poly_flat[1::2]
    w_px = max(xs) - min(xs)
    h_px = max(ys) - min(ys)
    w_mm = w_px * mmpx_x
    h_mm = h_px * mmpx_y
    # Shoelace 로 폴리곤 픽셀 면적 → bbox 대비 채움 비율
    n = len(xs)
    s = 0.0
    for i in range(n):
        j = (i + 1) % n
        s += xs[i] * ys[j] - xs[j] * ys[i]
    poly_px = abs(s) / 2.0
    bbox_px = w_px * h_px
    fill = (poly_px / bbox_px * 100.0) if bbox_px > 0 else None
    return {
        'bbox_width_mm': round(w_mm, 2) if w_mm else None,
        'bbox_height_mm': round(h_mm, 2) if h_mm else None,
        'aspect_wh': round(w_mm / h_mm, 3) if (w_mm and h_mm > 0) else None,
        'polygon_fill_pct': round(fill, 2) if fill is not None else None,
    }


def _compute_frame_area_ratio(defects, coord_system, polygons=None, image_size=None,
                                 visible_pipe_area_mm2=None):
    """프레임 단위 면적비 다중 척도.

    세 가지 척도 동시 산출:
      A. screen_pixel_ratio  — 화면 폴리곤 픽셀 ÷ 영상 픽셀 (육안 직관)
      B. visible_surface_ratio — 결함 표면적 ÷ 카메라 실측 가시 표면적 (PPNet 가시 영역)
      C. full_mesh_ratio (기존, defect_ratio_percent 키) — 결함 ÷ 전개도 메시 전체

    Args:
        defects: [{area_mm2}, ...]
        coord_system: 전개도 좌표계 dict (기존 척도 C 용)
        polygons: 원본 프레임 폴리곤 좌표 list of [x1,y1,x2,y2,...] (척도 A 용)
        image_size: (image_w, image_h) tuple (척도 A 용)
        visible_pipe_area_mm2: PPNet 가시 영역 mm² (척도 B 용)
    """
    full_mesh_mm2 = float(coord_system.get('x_range_mm', 0) * coord_system.get('y_range_mm', 0))
    total_defect_area_mm2 = float(sum(float(d.get('area_mm2', 0) or 0) for d in defects))

    result = {
        # 기존 호환 키 (전개도 메시 전체 분모)
        'visible_pipe_area_mm2': round(full_mesh_mm2, 1),
        'visible_pipe_area_cm2': round(full_mesh_mm2 / 100.0, 2),
        'total_defect_area_mm2': round(total_defect_area_mm2, 1),
        'total_defect_area_cm2': round(total_defect_area_mm2 / 100.0, 2),
        'defect_ratio_percent': round((total_defect_area_mm2 / full_mesh_mm2 * 100.0)
                                       if full_mesh_mm2 > 0 else 0.0, 4),
    }

    # A. 화면 픽셀 비율 (모든 모드 공통, 가장 직관적)
    if polygons and image_size:
        img_w, img_h = image_size
        total_poly_px = 0.0
        for p in polygons:
            if not p or len(p) < 6:
                continue
            xs = p[0::2]
            ys = p[1::2]
            n = len(xs)
            s = 0.0
            for i in range(n):
                j = (i + 1) % n
                s += xs[i] * ys[j] - xs[j] * ys[i]
            total_poly_px += abs(s) / 2.0
        frame_px = float(img_w * img_h)
        if frame_px > 0:
            result['screen_pixel_defect_px'] = int(round(total_poly_px))
            result['screen_pixel_frame_px'] = int(frame_px)
            result['screen_pixel_ratio_pct'] = round(total_poly_px / frame_px * 100.0, 4)

    # B. 가시 표면 비율 (PPNet 가시 영역 분모)
    if visible_pipe_area_mm2 and visible_pipe_area_mm2 > 0:
        result['visible_surface_pipe_area_mm2'] = round(float(visible_pipe_area_mm2), 1)
        result['visible_surface_pipe_area_cm2'] = round(float(visible_pipe_area_mm2) / 100.0, 2)
        result['visible_surface_ratio_pct'] = round(
            total_defect_area_mm2 / float(visible_pipe_area_mm2) * 100.0, 4)

    return result


@app.route('/api/sizing/initialize-depth', methods=['POST'])
def sizing_initialize_depth():
    try:
        model_type = (request.json or {}).get('model_type', 'MiDaS_small')
        depth_estimator = DepthEstimator.get_instance(model_type)
        depth_estimator._ensure_loaded()
        return jsonify({
            'success': True,
            'model_type': model_type,
            'device': str(DepthEstimator._device),
            'vram_mb': round(depth_estimator.get_vram_usage(), 1)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sizing/depth-map', methods=['POST'])
def sizing_depth_map():
    try:
        data = request.json or {}
        project_dir = data.get('project_dir')
        video_id = data.get('video_id')
        frame_number = int(data.get('frame_number', 0))
        if not project_dir or not video_id:
            return jsonify({'success': False, 'error': 'project_dir and video_id required'}), 400

        frame, actual_frame, total_frames, _ = _load_video_frame_for_sizing(project_dir, video_id, frame_number)
        depth_estimator = DepthEstimator.get_instance('MiDaS_small')
        depth_map = depth_estimator.estimate(frame, video_id=f'{project_dir}:{video_id}', frame_number=actual_frame)
        colorized = DepthEstimator.depth_to_colorized(depth_map)

        return jsonify({
            'success': True,
            'frame_number': actual_frame,
            'total_frames': total_frames,
            'colorized_preview': _encode_image_base64(colorized),
            'depth_map_png': DepthEstimator.depth_to_base64_png(depth_map)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sizing/calculate', methods=['POST'])
def sizing_calculate():
    try:
        data = request.json or {}
        project_dir = data.get('project_dir')
        video_id = data.get('video_id')
        frame_number = int(data.get('frame_number', 0))
        pipe_diameter_mm = float(data.get('pipe_diameter_mm', 300))
        defects = data.get('defects') or []
        use_depth = bool(data.get('use_depth', True))
        if not project_dir or not video_id:
            return jsonify({'success': False, 'error': 'project_dir and video_id required'}), 400

        frame, actual_frame, _, _ = _load_video_frame_for_sizing(project_dir, video_id, frame_number)
        cache_key = f'{project_dir}:{video_id}'
        vp = _resolve_vp_from_request(data, frame, cache_key)
        calibrator = PipeSizeCalibrator(pipe_diameter_mm, vp['vp_x'], vp['vp_y'], frame.shape[1], frame.shape[0])

        depth_map = None
        depth_preview = None
        if use_depth:
            depth_estimator = DepthEstimator.get_instance('MiDaS_small')
            depth_map = depth_estimator.estimate(frame, video_id=cache_key, frame_number=actual_frame)
            depth_preview = _encode_image_base64(DepthEstimator.depth_to_colorized(depth_map))

        measurements = _measure_defects(defects, calibrator, depth_map=depth_map, vp=vp)
        if project_dir:
            try:
                SizingResultManager.save_results(project_dir, video_id, {'frame_number': actual_frame, 'measurements': measurements})
            except Exception:
                pass

        return jsonify({
            'success': True,
            'frame_number': actual_frame,
            'vp': vp,
            'measurements': measurements,
            'depth_map_preview': depth_preview,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sizing/area-ratio', methods=['POST'])
def sizing_area_ratio():
    try:
        data = request.json or {}
        pipe_diameter_mm = float(data.get('pipe_diameter_mm', 300))
        section_length_mm = float(data.get('section_length_mm', 6000))
        measurements = data.get('measurements') or []
        calc = PipeAreaRatioCalculator()
        result = calc.calculate_section_ratio(pipe_diameter_mm, section_length_mm, defect_measurements=measurements)
        return jsonify({'success': True, **result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sizing/unwrap', methods=['POST'])
def sizing_unwrap():
    try:
        data = request.json or {}
        project_dir = data.get('project_dir')
        video_id = data.get('video_id')
        frame_number = int(data.get('frame_number', 0))
        pipe_diameter_mm = float(data.get('pipe_diameter_mm', 300))
        output_width = int(data.get('output_width', 800))
        output_height = int(data.get('output_height', 600))
        defects = data.get('defects') or []
        if not project_dir or not video_id:
            return jsonify({'success': False, 'error': 'project_dir and video_id required'}), 400

        frame, actual_frame, _, _ = _load_video_frame_for_sizing(project_dir, video_id, frame_number)
        vp = _resolve_vp_from_request(data, frame, f'{project_dir}:{video_id}')
        unwrapper = PipeUnwrapper(vp['vp_x'], vp['vp_y'], frame.shape[1], frame.shape[0], pipe_diameter_mm, output_width, output_height)
        unwrapped = unwrapper.unwrap(frame)
        coord = unwrapper.get_coordinate_system()

        result_defects = []
        for defect in defects:
            polygon_flat = _normalize_defect_polygon(defect)
            if not polygon_flat:
                continue
            unwrapped_polygon = unwrapper.transform_polygon(polygon_flat)
            area = unwrapper.calculate_unwrapped_area(unwrapped_polygon)
            result_defects.append({
                'index': defect.get('index', defect.get('annotation_index', -1)),
                'category': defect.get('category', defect.get('label', 'unknown')),
                'unwrapped_polygon': unwrapped_polygon,
                **area
            })

        frame_area_ratio = _compute_frame_area_ratio(result_defects, coord)
        return jsonify({
            'success': True,
            'frame_number': actual_frame,
            'vp': vp,
            'unwrapped_image': _encode_image_base64(unwrapped),
            'defects': result_defects,
            'coordinate_system': coord,
            'frame_area_ratio': frame_area_ratio
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/sizing/depth-unwrap', methods=['POST'])
def sizing_depth_unwrap():
    try:
        data = request.json or {}
        project_dir = data.get('project_dir')
        video_id = data.get('video_id')
        frame_number = int(data.get('frame_number', 0))
        pipe_diameter_mm = float(data.get('pipe_diameter_mm', 300))
        output_width = int(data.get('output_width', 800))
        output_height = int(data.get('output_height', 600))
        defects = data.get('defects') or []
        if not project_dir or not video_id:
            return jsonify({'success': False, 'error': 'project_dir and video_id required'}), 400

        frame, actual_frame, _, _ = _load_video_frame_for_sizing(project_dir, video_id, frame_number)
        cache_key = f'{project_dir}:{video_id}'
        vp = _resolve_vp_from_request(data, frame, cache_key)
        depth_estimator = DepthEstimator.get_instance('MiDaS_small')
        depth_map = depth_estimator.estimate(frame, video_id=cache_key, frame_number=actual_frame)
        calibrator = PipeSizeCalibrator(pipe_diameter_mm, vp['vp_x'], vp['vp_y'], frame.shape[1], frame.shape[0])
        unwrapper = DepthAwarePipeUnwrapper(vp['vp_x'], vp['vp_y'], frame.shape[1], frame.shape[0], pipe_diameter_mm, output_width, output_height)
        unwrapper.set_depth_map(depth_map, calibrator)
        unwrapped = unwrapper.unwrap(frame)
        depth_overlay = unwrapper.unwrap_depth(depth_map)
        coord = unwrapper.get_coordinate_system()

        result_defects = []
        for defect in defects:
            polygon_flat = _normalize_defect_polygon(defect)
            if not polygon_flat:
                continue
            unwrapped_polygon = unwrapper.transform_polygon(polygon_flat)
            area = unwrapper.calculate_unwrapped_area(unwrapped_polygon)
            result_defects.append({
                'index': defect.get('index', defect.get('annotation_index', -1)),
                'category': defect.get('category', defect.get('label', 'unknown')),
                'unwrapped_polygon': unwrapped_polygon,
                **area
            })

        y_profile = []
        if getattr(unwrapper, 'mm_per_px_y_array', None) is not None:
            arr = unwrapper.mm_per_px_y_array
            step = max(1, len(arr)//20)
            y_profile = [round(float(v), 4) for v in arr[::step]]

        frame_area_ratio = _compute_frame_area_ratio(result_defects, coord)
        return jsonify({
            'success': True,
            'frame_number': actual_frame,
            'vp': vp,
            'unwrapped_image': _encode_image_base64(unwrapped),
            'depth_overlay': _encode_image_base64(depth_overlay),
            'defects': result_defects,
            'coordinate_system': coord,
            'frame_area_ratio': frame_area_ratio,
            'y_scale_profile': y_profile
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════
#  Quick Sizing — 프로젝트 없이 임의 영상/프레임으로 사이징·면적비 산출
# ═══════════════════════════════════════════════════════════════════════
import tempfile
import uuid
import shutil
from pathlib import Path as _Path

QUICK_SIZING_DIR = _Path(tempfile.gettempdir()) / 'quick_sizing'
QUICK_SIZING_DIR.mkdir(parents=True, exist_ok=True)
QUICK_SIZING_SNAPSHOT_DIR = QUICK_SIZING_DIR / 'snapshots'
QUICK_SIZING_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
QUICK_SIZING_SESSIONS = {}  # token -> {'path', 'fps', 'frame_count', 'width', 'height', 'created_at'}
QUICK_SIZING_TTL_SEC = 6 * 3600  # 6시간 (영상 본체. 스냅샷은 로그와 함께 보존)
QUICK_SIZING_LOG_PATH = QUICK_SIZING_DIR / 'analyze_log.jsonl'
_quick_sizing_log_lock = threading.Lock()


def _quick_sizing_save_snapshot(frame_bgr, token, frame_number):
    """분석 시점 프레임을 디스크에 저장. 이미 있으면 재사용. 상대 파일명 반환."""
    try:
        fname = f'{token}_{int(frame_number):06d}.jpg'
        fpath = QUICK_SIZING_SNAPSHOT_DIR / fname
        if not fpath.exists():
            cv2.imwrite(str(fpath), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return fname
    except Exception as e:
        print(f'[quick-sizing] snapshot save failed: {e}')
        return None


def _quick_sizing_log(record: dict) -> None:
    """JSONL 한 줄 추가 (스레드 안전). 실패해도 분석 응답은 영향 없음."""
    try:
        with _quick_sizing_log_lock:
            with open(QUICK_SIZING_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:
        print(f'[quick-sizing] log write failed: {e}')


def _compute_accuracy(measurements, unwrap_defects, ground_truth):
    """기준값과 측정값 비교 → 폴리곤별 오차 + 전체 MAPE.

    ground_truth: [{area_mm2?, width_mm?, height_mm?, label?}, ...] (인덱스 매칭)
    """
    if not ground_truth:
        return None
    rows = []
    abs_errs = []
    for i, gt in enumerate(ground_truth):
        if not isinstance(gt, dict):
            continue
        m = measurements[i] if i < len(measurements) else {}
        u = unwrap_defects[i] if unwrap_defects and i < len(unwrap_defects) else {}
        row = {'index': i, 'gt': gt}
        # Calibrator 측정
        meas_area = m.get('real_area_mm2')
        if meas_area is not None and gt.get('area_mm2'):
            err = (meas_area - gt['area_mm2']) / gt['area_mm2'] * 100.0
            row['calibrator_area_mm2'] = meas_area
            row['calibrator_error_pct'] = round(err, 2)
            abs_errs.append(abs(err))
        # 전개도 측정
        uw_area = u.get('area_mm2')
        if uw_area is not None and gt.get('area_mm2'):
            err = (uw_area - gt['area_mm2']) / gt['area_mm2'] * 100.0
            row['unwrap_area_mm2'] = uw_area
            row['unwrap_error_pct'] = round(err, 2)
        rows.append(row)
    mape = round(sum(abs_errs) / len(abs_errs), 2) if abs_errs else None
    return {
        'rows': rows,
        'mape_pct': mape,
        'n_compared': len(abs_errs),
    }


def _quick_sizing_evict_old():
    """메모리 세션만 만료. 디스크 파일은 보존 (이력 복원용)."""
    now = time.time()
    expired = [t for t, s in QUICK_SIZING_SESSIONS.items()
               if now - s.get('created_at', 0) > QUICK_SIZING_TTL_SEC]
    for t in expired:
        QUICK_SIZING_SESSIONS.pop(t, None)


def _quick_sizing_attach_session(video_path):
    """디스크 영상 파일에 새 세션 부착 (메타데이터 채움)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f'video missing: {video_path}')
    ext = os.path.splitext(video_path)[1].lower()
    is_image = ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
    token = uuid.uuid4().hex
    if is_image:
        img = cv2.imread(video_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('cannot read image')
        h, w = img.shape[:2]
        meta = {'path': video_path, 'kind': 'image', 'fps': 0.0,
                'frame_count': 1, 'width': w, 'height': h, 'created_at': time.time()}
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError('cannot open video')
        meta = {'path': video_path, 'kind': 'video',
                'fps': float(cap.get(cv2.CAP_PROP_FPS) or 0),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                'created_at': time.time()}
        cap.release()
    QUICK_SIZING_SESSIONS[token] = meta
    return token, meta


def _quick_sizing_get_session(token):
    sess = QUICK_SIZING_SESSIONS.get(token)
    if not sess:
        raise FileNotFoundError(f'unknown token: {token}')
    if not os.path.exists(sess['path']):
        QUICK_SIZING_SESSIONS.pop(token, None)
        raise FileNotFoundError(f'session file missing: {token}')
    return sess


def _quick_sizing_read_frame(sess, frame_number):
    cap = cv2.VideoCapture(sess['path'])
    if not cap.isOpened():
        raise RuntimeError(f'failed to open: {sess["path"]}')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or sess.get('frame_count', 0)
    frame_number = max(0, min(int(frame_number), max(0, total - 1)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    if (not ok or frame is None) and frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        frame_number = 0
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f'failed to read frame {frame_number}')
    return frame, frame_number


@app.route('/api/quick-sizing/upload', methods=['POST'])
def quick_sizing_upload():
    """영상/이미지 업로드 → 임시 토큰 반환.

    multipart form:
      file: video (mp4/avi/...) 또는 image (jpg/png)

    응답: { token, kind: 'video'|'image', frame_count, fps, width, height }
    """
    try:
        _quick_sizing_evict_old()
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'multipart file required'}), 400
        f = request.files['file']
        if not f or f.filename == '':
            return jsonify({'success': False, 'error': 'empty filename'}), 400

        suffix = os.path.splitext(f.filename)[1].lower() or '.mp4'
        token = uuid.uuid4().hex
        save_path = str(QUICK_SIZING_DIR / f'{token}{suffix}')
        f.save(save_path)

        is_image = suffix in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
        if is_image:
            img = cv2.imread(save_path, cv2.IMREAD_COLOR)
            if img is None:
                os.remove(save_path)
                return jsonify({'success': False, 'error': 'invalid image'}), 400
            h, w = img.shape[:2]
            meta = {'path': save_path, 'kind': 'image', 'fps': 0.0,
                    'frame_count': 1, 'width': w, 'height': h, 'created_at': time.time()}
        else:
            cap = cv2.VideoCapture(save_path)
            if not cap.isOpened():
                cap.release()
                os.remove(save_path)
                return jsonify({'success': False, 'error': 'cannot open video'}), 400
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            meta = {'path': save_path, 'kind': 'video', 'fps': fps,
                    'frame_count': n, 'width': w, 'height': h, 'created_at': time.time()}
        QUICK_SIZING_SESSIONS[token] = meta
        return jsonify({
            'success': True,
            'token': token,
            'kind': meta['kind'],
            'frame_count': meta['frame_count'],
            'fps': meta['fps'],
            'width': meta['width'],
            'height': meta['height'],
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/quick-sizing/frame', methods=['POST'])
def quick_sizing_frame():
    """토큰 + 프레임 번호 → 프레임 JPEG base64 + 자동 VP 검출.

    JSON: { token, frame_number, detect_vp?: bool }
    응답: { success, frame_base64, frame_number, vp?: {...}, vp_radial?, vp_darkest? }
    """
    try:
        data = request.json or {}
        token = data.get('token')
        if not token:
            return jsonify({'success': False, 'error': 'token required'}), 400
        sess = _quick_sizing_get_session(token)

        if sess['kind'] == 'image':
            frame = cv2.imread(sess['path'], cv2.IMREAD_COLOR)
            actual = 0
        else:
            frame, actual = _quick_sizing_read_frame(sess, int(data.get('frame_number', 0)))

        out = {
            'success': True,
            'frame_number': actual,
            'frame_base64': _encode_image_base64(frame),
            'width': int(frame.shape[1]),
            'height': int(frame.shape[0]),
        }
        if bool(data.get('detect_vp', True)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_masked = vp_detector._mask_osd(gray)
            r_radial = vp_detector._detect_radial_convergence(gray_masked)
            r_dark = vp_detector._detect_gaussian_darkest(gray_masked)
            best = r_radial if r_radial['confidence'] >= r_dark['confidence'] else r_dark
            cache_key = f'quick:{token}'
            vp_detector._cache[cache_key] = {'vp': best, 'timestamp': time.time()}
            out['vp'] = best
            out['vp_radial'] = r_radial
            out['vp_darkest'] = r_dark
        return jsonify(out)
    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def _quick_sizing_analyze_impl(data):
    """Quick Sizing 분석 핵심 로직. (response_dict, status_code) 반환.

    HTTP 라우트 + batch 양쪽에서 재사용.
    """
    try:
        token = data.get('token')
        if not token:
            return ({'success': False, 'error': 'token required'}, 400)
        sess = _quick_sizing_get_session(token)

        pipe_diameter_mm = float(data.get('pipe_diameter_mm', 300))
        section_length_mm = float(data.get('section_length_mm', 6000))
        use_depth = bool(data.get('use_depth', True))
        # 전개 방식: 'none' | 'vp' | 'ppnet' (옛 include_unwrap 호환)
        unwrap_method = (data.get('unwrap_method') or '').lower()
        if not unwrap_method:
            unwrap_method = 'vp' if data.get('include_unwrap', True) else 'none'
        if unwrap_method not in ('none', 'vp', 'ppnet'):
            return ({'success': False, 'error': f'invalid unwrap_method: {unwrap_method}'}, 400)
        out_w = int(data.get('output_width', 800))
        out_h = int(data.get('output_height', 600))
        # PPNet 3D 전개에서 메시가 cover 하는 축 방향 최대 깊이.
        # 너무 작으면 화면 중앙(원근 먼 쪽) 결함이 메시 밖으로 잘려서 area_px=0.
        max_depth_mm = float(data.get('max_depth_mm', 1000))
        # 카메라 intrinsic override (캘리브레이션). PPNet 3D 모드에만 영향.
        camera_f_override = data.get('camera_f')
        if camera_f_override is not None:
            try:
                camera_f_override = float(camera_f_override)
                if camera_f_override <= 0:
                    camera_f_override = None
            except (TypeError, ValueError):
                camera_f_override = None
        ground_truth = data.get('ground_truth') or None  # [{area_mm2?, label?}, ...]
        log_enabled = data.get('log_enabled', True)
        log_note = data.get('log_note') or ''

        raw_polys = data.get('polygons') or data.get('defects') or []
        defects = []
        for i, p in enumerate(raw_polys):
            if isinstance(p, dict):
                poly = _normalize_defect_polygon(p)
                label = p.get('label') or p.get('category') or 'defect'
            elif isinstance(p, list):
                poly = p if len(p) >= 6 else None
                label = 'defect'
            else:
                poly = None
                label = 'defect'
            if poly:
                defects.append({'index': i, 'label': label, 'polygon': poly})

        if sess['kind'] == 'image':
            frame = cv2.imread(sess['path'], cv2.IMREAD_COLOR)
            actual = 0
        else:
            frame, actual = _quick_sizing_read_frame(sess, int(data.get('frame_number', 0)))

        # 복원용: 영상 파일 경로 (디스크에 보존되어 있어 재추출 가능)
        video_path_abs = sess.get('path')

        cache_key = f'quick:{token}'
        vp = _resolve_vp_from_request(data, frame, cache_key)
        # Calibrator: camera_f 가 있으면 광학 모델, 없으면 이미지 크기로 HD/FHD 기본값 사용
        calibrator = PipeSizeCalibrator(pipe_diameter_mm, vp['vp_x'], vp['vp_y'],
                                          frame.shape[1], frame.shape[0],
                                          camera_f=camera_f_override)

        depth_map = None
        depth_preview = None
        if use_depth:
            try:
                depth_estimator = DepthEstimator.get_instance('MiDaS_small')
                depth_map = depth_estimator.estimate(frame, video_id=cache_key, frame_number=actual)
                depth_preview = _encode_image_base64(DepthEstimator.depth_to_colorized(depth_map))
            except Exception as de:
                print(f'[quick-sizing] depth estimation failed: {de}')

        measurements = _measure_defects(defects, calibrator, depth_map=depth_map, vp=vp)

        # 면적비 (section 기반)
        area_ratio_full = None
        try:
            calc = PipeAreaRatioCalculator()
            area_ratio_full = calc.calculate_section_ratio(
                pipe_diameter_mm, section_length_mm,
                defect_measurements=[m for m in measurements if not m.get('error')]
            )
        except Exception as are:
            print(f'[quick-sizing] area ratio failed: {are}')

        # 전개도 + 프레임 단위 면적비
        unwrap_out = None
        frame_area_ratio = None
        ppnet_pose = None

        if unwrap_method == 'vp' and defects:
            try:
                unwrapper = PipeUnwrapper(vp['vp_x'], vp['vp_y'],
                                            frame.shape[1], frame.shape[0],
                                            pipe_diameter_mm, out_w, out_h)
                unwrapped = unwrapper.unwrap(frame)
                coord = unwrapper.get_coordinate_system()
                uw_defs = []
                mmpx_x = coord.get('mm_per_px_x', 1.0)
                mmpx_y = coord.get('mm_per_px_y', 1.0)
                for d in defects:
                    pts = d['polygon']
                    uw_poly = unwrapper.transform_polygon(pts)
                    area = unwrapper.calculate_unwrapped_area(uw_poly)
                    # 전개도 좌표계에서 bbox 산출 (직사각형 종횡비 검증용)
                    bbox = _bbox_of_flat_polygon(uw_poly, mmpx_x, mmpx_y)
                    uw_defs.append({'index': d['index'], 'label': d['label'],
                                     'unwrapped_polygon': uw_poly, **area, **bbox})
                frame_area_ratio = _compute_frame_area_ratio(
                    uw_defs, coord,
                    polygons=[d['polygon'] for d in defects],
                    image_size=(frame.shape[1], frame.shape[0]),
                )
                unwrap_out = {
                    'method': 'vp',
                    'unwrapped_image': _encode_image_base64(unwrapped),
                    'coordinate_system': coord,
                    'defects': uw_defs,
                }
            except Exception as ue:
                print(f'[quick-sizing] vp unwrap failed: {ue}')

        elif unwrap_method == 'ppnet':
            try:
                from gnu_mapping import GNUMappingEngine
                ppnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'ppnet.pt')
                if not os.path.exists(ppnet_path):
                    return ({'success': False, 'error': 'PPNet 가중치 없음: weights/ppnet.pt'}, 400)

                # 폴리곤 → 마스크 변환
                defect_masks = []
                for d in defects:
                    poly = d['polygon']
                    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    pts = np.array([[int(poly[i]), int(poly[i+1])]
                                     for i in range(0, len(poly), 2)], dtype=np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(mask, [pts], 255)
                        defect_masks.append({'label': d['label'], 'mask': mask})

                # 사용자가 수동 자세를 지정했으면 PPNet 추론 스킵
                use_manual_pose = bool(data.get('use_manual_pose', False))
                pose_override = None
                vp_in = data.get('vp') or {}
                if use_manual_pose and 'vp_x' in vp_in and 'vp_y' in vp_in:
                    pose_override = (
                        float(vp_in['vp_x']),
                        float(vp_in['vp_y']),
                        float(vp_in.get('angle', 0.0)),
                        float(vp_in.get('step', 0.0)),
                    )

                engine = GNUMappingEngine(ppnet_model_path=ppnet_path,
                                            pipe_diameter_mm=pipe_diameter_mm,
                                            water=False,
                                            pixel_per_mm=10.0,
                                            max_depth_mm=max_depth_mm)
                camera_override = None
                if camera_f_override is not None:
                    camera_override = {'f': camera_f_override}
                gnu_result = engine.process_frame(frame, defect_masks or None,
                                                    include_depth_map=use_depth,
                                                    pose_override=pose_override,
                                                    camera_override=camera_override)
                ppnet_pose = gnu_result.get('pose')
                # 자세 출처 메타 (manual vs ppnet)
                if ppnet_pose:
                    ppnet_pose = {**ppnet_pose}
                    ppnet_pose.setdefault('source', 'ppnet')
                coord = gnu_result.get('coordinate_system') or {}
                # 응답 스키마를 VP 모드와 동일하게 정규화
                uw_defs = []
                for ud in gnu_result.get('unwrapped_defects', []) or []:
                    uw_defs.append({
                        'label': ud.get('label'),
                        'area_px': ud.get('area_px'),
                        'area_mm2': ud.get('area_mm2'),
                        'area_cm2': ud.get('area_cm2'),
                        'area_ratio_pct': ud.get('area_ratio_pct'),
                        'area_ratio_visible_pct': ud.get('area_ratio_visible_pct'),
                        'weighted_ratio_pct': ud.get('weighted_ratio_pct'),
                        'weighted_area_mm2': ud.get('weighted_area_mm2'),
                        'avg_camera_distance_mm': ud.get('avg_camera_distance_mm'),
                        'bbox_width_mm': ud.get('bbox_width_mm'),
                        'bbox_height_mm': ud.get('bbox_height_mm'),
                        'aspect_wh': ud.get('aspect_wh'),
                        'polygon_fill_pct': ud.get('polygon_fill_pct'),
                    })
                # 프레임 면적비 (다중 척도: 화면 픽셀, 가시 표면, 메시 전체)
                frame_area_ratio = _compute_frame_area_ratio(
                    [{'area_mm2': d.get('area_mm2') or 0} for d in uw_defs], coord,
                    polygons=[d['polygon'] for d in defects],
                    image_size=(frame.shape[1], frame.shape[0]),
                    visible_pipe_area_mm2=gnu_result.get('visible_unwrap_mm2'),
                )
                unwrap_out = {
                    'method': 'ppnet',
                    'unwrapped_image': (gnu_result.get('unwrapped_overlay_b64')
                                          or gnu_result.get('unwrapped_rgb_b64')),
                    'coordinate_system': coord,
                    'defects': uw_defs,
                    'visible_coverage_pct': gnu_result.get('visible_coverage_pct'),
                    'visible_unwrap_mm2': gnu_result.get('visible_unwrap_mm2'),
                    'unwrap_total_mm2': gnu_result.get('unwrap_total_mm2'),
                }
                if not depth_preview and gnu_result.get('depth_heatmap_b64'):
                    depth_preview = gnu_result['depth_heatmap_b64']
                # PPNet 자세를 vp 정보로 보강 (UI 표시용)
                if ppnet_pose:
                    vp = {
                        **vp,
                        'vp_x': ppnet_pose['vp_x'],
                        'vp_y': ppnet_pose['vp_y'],
                        'angle': ppnet_pose['angle'],
                        'step': ppnet_pose['step'],
                        'method': 'ppnet',
                        'confidence': 1.0,
                    }
            except Exception as pe:
                import traceback
                traceback.print_exc()
                print(f'[quick-sizing] ppnet unwrap failed: {pe}')

        # 'none' 모드 또는 unwrap 실패해도 화면 픽셀 비율은 계산
        if frame_area_ratio is None and defects:
            frame_area_ratio = _compute_frame_area_ratio(
                [], {},
                polygons=[d['polygon'] for d in defects],
                image_size=(frame.shape[1], frame.shape[0]),
            )

        # 기준값 vs 측정값 → 오차
        uw_defs_for_acc = (unwrap_out or {}).get('defects') if unwrap_out else None
        accuracy = _compute_accuracy(measurements, uw_defs_for_acc, ground_truth)

        # JSONL 로그 (요청+요약 결과만, 큰 base64는 길이만 기록)
        if log_enabled:
            try:
                _quick_sizing_log({
                    'ts': time.time(),
                    'ts_iso': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()),
                    'token': token,
                    'session_kind': sess.get('kind'),
                    'note': log_note,
                    'request': {
                        'frame_number': int(data.get('frame_number', 0)),
                        'pipe_diameter_mm': pipe_diameter_mm,
                        'section_length_mm': section_length_mm,
                        'max_depth_mm': max_depth_mm,
                        'unwrap_method': unwrap_method,
                        'use_depth': use_depth,
                        'vp_input': data.get('vp'),
                        'polygons_n': len(defects),
                        'polygon_points': [len(d['polygon']) // 2 for d in defects],
                        # 폴리곤 좌표 전체 저장 — 추후 종횡비/회전 분석 등에 사용
                        'polygons': [d['polygon'] for d in defects],
                        'ground_truth': ground_truth,
                    },
                    'video_path': video_path_abs,        # 복원용 — 디스크 영상 경로
                    'session_kind': sess.get('kind'),
                    'result': {
                        'frame_number': actual,
                        'vp_resolved': vp,
                        'ppnet_pose': ppnet_pose,
                        'measurements': [{k: v for k, v in m.items()
                                          if k not in ('vp', 'center_px')} for m in measurements],
                        'area_ratio': area_ratio_full,
                        'frame_area_ratio': frame_area_ratio,
                        'unwrap_defects': uw_defs_for_acc,
                        'unwrap_coord': (unwrap_out or {}).get('coordinate_system'),
                        'accuracy': accuracy,
                        'depth_preview_size': len(depth_preview or ''),
                        'unwrap_image_size': len((unwrap_out or {}).get('unwrapped_image') or ''),
                    },
                })
            except Exception as le:
                print(f'[quick-sizing] log compose failed: {le}')

        return ({
            'success': True,
            'frame_number': actual,
            'video_path': video_path_abs,
            'vp': vp,
            'ppnet_pose': ppnet_pose,
            'unwrap_method': unwrap_method,
            'pipe_diameter_mm': pipe_diameter_mm,
            'section_length_mm': section_length_mm,
            'max_depth_mm': max_depth_mm,
            'camera_f_used': camera_f_override,  # 사용된 카메라 f (None이면 기본 프리셋)
            'measurements': measurements,
            'area_ratio': area_ratio_full,
            'frame_area_ratio': frame_area_ratio,
            'unwrap': unwrap_out,
            'depth_preview': depth_preview,
            'accuracy': accuracy,
        }, 200)
    except FileNotFoundError as e:
        return ({'success': False, 'error': str(e)}, 404)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ({'success': False, 'error': str(e)}, 500)


@app.route('/api/quick-sizing/analyze', methods=['POST'])
def quick_sizing_analyze():
    """HTTP 래퍼 — 단일 프레임 분석. (호환성 유지)"""
    body, status = _quick_sizing_analyze_impl(request.json or {})
    return jsonify(body), status


@app.route('/api/quick-sizing/calibrate-f', methods=['POST'])
def quick_sizing_calibrate_f():
    """카메라 초점거리(f) 자동 캘리브레이션.

    GT 면적이 입력된 폴리곤(들) 의 측정값과 기준값이 일치하도록 f 를 도출.

    PPNet 3D 면적 = 실제 면적 × (f_가정 / f_실제)²
    → f_실제 = f_가정 × √(GT / measured)

    1-2회 반복으로 수렴.

    JSON: 일반 analyze 와 동일 (token, polygons, ground_truth, unwrap_method='ppnet' 권장 등)
    응답: { success, history: [...], recommended_f, fov_h_deg, final_measured, final_error_pct }
    """
    try:
        data = request.json or {}
        # unwrap_method 강제 ppnet (Calibrator/VP 는 f 영향 없음)
        data['unwrap_method'] = 'ppnet'
        data['log_enabled'] = False
        # GT 검증
        gt_list = data.get('ground_truth') or []
        gt_pairs = []  # (idx, gt_area_mm2)
        for i, g in enumerate(gt_list):
            if isinstance(g, dict) and g.get('area_mm2'):
                gt_pairs.append((i, float(g['area_mm2'])))
        if not gt_pairs:
            return jsonify({'success': False, 'error': 'ground_truth area 가 입력된 폴리곤이 최소 1개 필요'}), 400

        # 시작 f
        start_f = data.get('camera_f')
        if start_f is None:
            # 영상 크기로 기본 프리셋 결정
            from gnu_mapping import CAMERA_PARAMS, detect_resolution
            token = data.get('token')
            sess = _quick_sizing_get_session(token)
            w, h = sess.get('width', 1280), sess.get('height', 720)
            res_key = detect_resolution(w, h)
            start_f = float(CAMERA_PARAMS[res_key]['f'])

        history = []
        current_f = float(start_f)
        max_iter = 6  # 직접 공식이지만 비선형 인자(자세, max_depth 등) 때문에 1-2회로 안 수렴할 수 있음
        last_measured = None

        for it in range(max_iter):
            data['camera_f'] = current_f
            body, status = _quick_sizing_analyze_impl(data)
            if not body.get('success'):
                return jsonify({'success': False, 'error': body.get('error') or 'analyze failed',
                                  'iteration': it}), 500
            # 측정 면적 — GT 가 있는 폴리곤들의 평균 비율 (geometric)
            uw_defs = ((body.get('unwrap') or {}).get('defects')) or []
            ratios = []
            details = []
            for idx, gt_area in gt_pairs:
                if idx < len(uw_defs) and uw_defs[idx].get('area_mm2'):
                    measured = float(uw_defs[idx]['area_mm2'])
                    ratios.append(measured / gt_area)
                    details.append({'idx': idx, 'measured': measured, 'gt': gt_area,
                                     'ratio': round(measured / gt_area, 4)})
            if not ratios:
                return jsonify({'success': False, 'error': 'measured area unavailable',
                                  'iteration': it, 'history': history}), 500

            # 기하 평균 비율 (여러 GT 시 균형)
            import math
            geom_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            err_pct = (geom_ratio - 1.0) * 100
            history.append({
                'iter': it, 'f_used': round(current_f, 2),
                'geom_ratio': round(geom_ratio, 4),
                'err_pct': round(err_pct, 2),
                'details': details,
            })
            last_measured = details

            # 수렴 판정
            if abs(err_pct) < 1.0:
                break
            # 다음 f 추정
            current_f = current_f * math.sqrt(1.0 / geom_ratio)

        # FOV 계산 (수평)
        w = sess.get('width', 1280) if 'sess' in dir() else 1280
        import math
        fov_h = math.degrees(2 * math.atan(w / (2 * current_f)))

        return jsonify({
            'success': True,
            'history': history,
            'recommended_f': round(current_f, 2),
            'start_f': round(start_f, 2),
            'fov_h_deg': round(fov_h, 1),
            'final_details': last_measured,
            'iterations': len(history),
        })
    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e)}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/quick-sizing/batch-analyze', methods=['POST'])
def quick_sizing_batch_analyze():
    """동일 폴리곤을 여러 프레임에 적용 → 일관성 비교.

    요청 JSON: {
      token, frame_numbers: [int...],
      polygons, pipe_diameter_mm, section_length_mm, max_depth_mm,
      unwrap_method, use_depth, use_manual_pose, vp,
      ground_truth, include_unwrap_image (기본 false)
    }
    응답: { success, count, frames: [...], summary: {pose_stability, per_polygon} }
    """
    try:
        data = request.json or {}
        frames = data.get('frame_numbers') or []
        if not isinstance(frames, list) or not frames:
            return jsonify({'success': False, 'error': 'frame_numbers required (non-empty list)'}), 400
        include_uw_img = bool(data.get('include_unwrap_image', False))

        per_frame = []
        for fn in frames:
            try:
                d = {**data, 'frame_number': int(fn)}
                d.pop('frame_numbers', None)
                d['log_enabled'] = False  # batch 중 로깅 안 함
                body, status = _quick_sizing_analyze_impl(d)
            except Exception as e:
                per_frame.append({'frame_number': fn, 'success': False, 'error': str(e)})
                continue
            if not body.get('success'):
                per_frame.append({'frame_number': fn, 'success': False,
                                   'error': body.get('error')})
                continue
            # 슬림화 (큰 이미지 base64 제외)
            slim_uw = None
            if body.get('unwrap'):
                uw = body['unwrap']
                slim_uw = {
                    'defects': uw.get('defects'),
                    'coordinate_system': uw.get('coordinate_system'),
                    'visible_coverage_pct': uw.get('visible_coverage_pct'),
                }
                if include_uw_img:
                    slim_uw['unwrapped_image'] = uw.get('unwrapped_image')
            per_frame.append({
                'success': True,
                'frame_number': body.get('frame_number'),
                'vp': body.get('vp'),
                'ppnet_pose': body.get('ppnet_pose'),
                'measurements': body.get('measurements'),
                'unwrap': slim_uw,
                'area_ratio': body.get('area_ratio'),
                'frame_area_ratio': body.get('frame_area_ratio'),
                'accuracy': body.get('accuracy'),
            })

        summary = _batch_summary(per_frame)
        return jsonify({
            'success': True,
            'count': len(per_frame),
            'frames': per_frame,
            'summary': summary,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def _batch_summary(per_frame):
    """프레임별 결과에서 통계 산출."""
    import statistics
    valid = [r for r in per_frame if r.get('success')]
    if not valid:
        return None

    # PPNet 자세 안정성
    angles = [r['ppnet_pose']['angle'] for r in valid
              if r.get('ppnet_pose') and r['ppnet_pose'].get('angle') is not None]
    steps = [r['ppnet_pose']['step'] for r in valid
             if r.get('ppnet_pose') and r['ppnet_pose'].get('step') is not None]
    vp_xs = [r['ppnet_pose']['vp_x'] for r in valid if r.get('ppnet_pose')]
    vp_ys = [r['ppnet_pose']['vp_y'] for r in valid if r.get('ppnet_pose')]

    def stats(arr):
        if not arr:
            return None
        return {
            'mean': round(statistics.mean(arr), 3),
            'std': round(statistics.stdev(arr), 3) if len(arr) >= 2 else 0.0,
            'min': round(min(arr), 3),
            'max': round(max(arr), 3),
        }

    pose_stab = None
    if angles and steps:
        pose_stab = {
            'n': len(angles),
            'vp_x': stats(vp_xs),
            'vp_y': stats(vp_ys),
            'angle': stats(angles),
            'step': stats(steps),
        }

    # 폴리곤별 면적 일관성
    n_polys = 0
    for r in valid:
        n_polys = max(n_polys, len(r.get('measurements') or []))

    per_poly = []
    for pi in range(n_polys):
        cal_areas = []
        uw_areas = []
        uw_aspects = []
        gt_area = None
        for r in valid:
            ms = r.get('measurements') or []
            if pi < len(ms) and not ms[pi].get('error'):
                a = ms[pi].get('real_area_mm2')
                if a is not None:
                    cal_areas.append(a)
            uw_defs = ((r.get('unwrap') or {}).get('defects')) or []
            if pi < len(uw_defs):
                a = uw_defs[pi].get('area_mm2')
                if a is not None:
                    uw_areas.append(a)
                asp = uw_defs[pi].get('aspect_wh')
                if asp is not None:
                    uw_aspects.append(asp)
            # GT
            acc = r.get('accuracy') or {}
            for row in acc.get('rows') or []:
                if row.get('index') == pi and gt_area is None:
                    gt = row.get('gt') or {}
                    if gt.get('area_mm2'):
                        gt_area = gt['area_mm2']

        entry = {'index': pi}
        if cal_areas:
            s = stats(cal_areas)
            s['cv_pct'] = round(s['std'] / s['mean'] * 100, 2) if s['mean'] else None
            if gt_area:
                s['mape_pct'] = round(
                    sum(abs(a - gt_area) / gt_area for a in cal_areas) /
                    len(cal_areas) * 100, 2)
            entry['calibrator'] = s
        if uw_areas:
            s = stats(uw_areas)
            s['cv_pct'] = round(s['std'] / s['mean'] * 100, 2) if s['mean'] else None
            if gt_area:
                s['mape_pct'] = round(
                    sum(abs(a - gt_area) / gt_area for a in uw_areas) /
                    len(uw_areas) * 100, 2)
            entry['unwrap'] = s
        if uw_aspects:
            entry['aspect_wh'] = stats(uw_aspects)
        if gt_area:
            entry['gt_area_mm2'] = gt_area
        per_poly.append(entry)

    return {'pose_stability': pose_stab, 'per_polygon': per_poly,
            'n_frames_valid': len(valid), 'n_frames_total': len(per_frame)}


@app.route('/api/quick-sizing/restore-session', methods=['POST'])
def quick_sizing_restore_session():
    """디스크에 남아 있는 영상 파일로 새 세션 부착.

    요청: { video_path: str }
    응답: { success, token, kind, frame_count, fps, width, height }
    """
    try:
        data = request.json or {}
        vpath = data.get('video_path')
        if not vpath:
            return jsonify({'success': False, 'error': 'video_path required'}), 400
        # 보안: QUICK_SIZING_DIR 하위만 허용
        try:
            vpath_resolved = str(_Path(vpath).resolve())
            qs_root = str(QUICK_SIZING_DIR.resolve())
            if not vpath_resolved.startswith(qs_root):
                return jsonify({'success': False, 'error': 'path outside quick_sizing dir'}), 403
        except Exception:
            return jsonify({'success': False, 'error': 'invalid path'}), 400
        if not os.path.exists(vpath_resolved):
            return jsonify({'success': False, 'error': 'video file no longer exists',
                             'requires_reupload': True}), 404

        # 이미 메모리에 같은 path 의 세션이 있으면 재사용
        for t, s in QUICK_SIZING_SESSIONS.items():
            if s.get('path') == vpath_resolved:
                return jsonify({'success': True, 'token': t, 'kind': s['kind'],
                                  'frame_count': s['frame_count'], 'fps': s['fps'],
                                  'width': s['width'], 'height': s['height'],
                                  'reused': True})

        token, meta = _quick_sizing_attach_session(vpath_resolved)
        return jsonify({'success': True, 'token': token, 'kind': meta['kind'],
                          'frame_count': meta['frame_count'], 'fps': meta['fps'],
                          'width': meta['width'], 'height': meta['height'],
                          'reused': False})
    except FileNotFoundError as e:
        return jsonify({'success': False, 'error': str(e), 'requires_reupload': True}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/quick-sizing/snapshot', methods=['GET'])
def quick_sizing_snapshot():
    """저장된 분석 시점 프레임 스냅샷 반환 (base64 JSON).

    Query: ?file=<filename> 또는 ?token=...&frame=...
    """
    try:
        fname = request.args.get('file') or request.args.get('path')
        if not fname:
            token = request.args.get('token')
            frame = request.args.get('frame')
            if not token or frame is None:
                return jsonify({'success': False, 'error': 'file or (token,frame) required'}), 400
            fname = f'{token}_{int(frame):06d}.jpg'
        if '/' in fname or '\\' in fname or '..' in fname:
            return jsonify({'success': False, 'error': 'invalid filename'}), 400
        path = QUICK_SIZING_SNAPSHOT_DIR / fname
        if not path.exists():
            return jsonify({'success': False, 'error': 'not found'}), 404
        with open(path, 'rb') as f:
            data = f.read()
        return jsonify({
            'success': True,
            'frame_base64': base64.b64encode(data).decode('utf-8'),
            'filename': fname,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/quick-sizing/log', methods=['GET'])
def quick_sizing_log_list():
    """JSONL 분석 로그 조회. 쿼리: limit, since (iso 또는 ts)."""
    try:
        limit = int(request.args.get('limit', 200))
        since = request.args.get('since')
        since_ts = None
        if since:
            try:
                since_ts = float(since)
            except ValueError:
                try:
                    since_ts = time.mktime(time.strptime(since[:19], '%Y-%m-%dT%H:%M:%S'))
                except Exception:
                    since_ts = None

        if not QUICK_SIZING_LOG_PATH.exists():
            return jsonify({'success': True, 'records': [], 'total': 0})

        records = []
        with open(QUICK_SIZING_LOG_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if since_ts is not None and r.get('ts', 0) < since_ts:
                        continue
                    records.append(r)
                except json.JSONDecodeError:
                    continue
        total = len(records)
        if limit > 0 and total > limit:
            records = records[-limit:]
        return jsonify({'success': True, 'records': records, 'total': total})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/quick-sizing/log', methods=['DELETE'])
def quick_sizing_log_clear():
    """JSONL 분석 로그 비우기. 영상/이미지/스냅샷 파일도 함께 일괄 정리.

    Query: ?keep_files=1 이면 파일은 남기고 로그만 비움.
    """
    try:
        keep_files = request.args.get('keep_files') in ('1', 'true', 'yes')
        if QUICK_SIZING_LOG_PATH.exists():
            QUICK_SIZING_LOG_PATH.unlink()

        n_videos = n_snap = 0
        if not keep_files:
            # 영상/이미지 파일 (확장자 무관, QUICK_SIZING_DIR 직속 파일만)
            for f in QUICK_SIZING_DIR.iterdir():
                if f.is_file() and f.name != 'analyze_log.jsonl':
                    try:
                        f.unlink(); n_videos += 1
                    except Exception:
                        pass
            # 스냅샷
            if QUICK_SIZING_SNAPSHOT_DIR.exists():
                for snap in QUICK_SIZING_SNAPSHOT_DIR.glob('*.jpg'):
                    try:
                        snap.unlink(); n_snap += 1
                    except Exception:
                        pass
            # 메모리 세션도 비움
            QUICK_SIZING_SESSIONS.clear()

        return jsonify({'success': True,
                          'videos_removed': n_videos,
                          'snapshots_removed': n_snap,
                          'kept_files': keep_files})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/quick-sizing/release', methods=['POST'])
def quick_sizing_release():
    """세션 명시적 정리. 메모리 세션만 비움. 디스크 영상은 보존 (이력 복원용).

    디스크 정리는 'log 비우기' 시 일괄 수행.
    """
    try:
        data = request.json or {}
        token = data.get('token')
        if not token:
            return jsonify({'success': False, 'error': 'token required'}), 400
        QUICK_SIZING_SESSIONS.pop(token, None)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/inference_raw', methods=['POST'])
def run_inference_raw():
    """Base64 이미지로 직접 SegFormer 추론 (pipe_survey용)"""
    global segformer_model, segformer_processor, segformer_device, ai_initialized

    with inference_lock:
        inference_stats['total_requests'] += 1
        inference_stats['active_requests'] += 1
        try:
            if not ai_initialized or segformer_model is None:
                return jsonify({'success': False, 'error': 'AI model not initialized'}), 400

            data = request.json
            img_b64 = data.get('image_base64')
            if not img_b64:
                return jsonify({'success': False, 'error': 'image_base64 required'}), 400

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = segformer_processor(images=frame_rgb, return_tensors="pt")
            inputs = {k: v.to(segformer_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = segformer_model(**inputs)
                logits = outputs.logits

            upsampled_logits = torch.nn.functional.interpolate(
                logits, size=frame.shape[:2], mode="bilinear", align_corners=False
            )
            predicted = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            bounding_boxes = extract_bounding_boxes_from_mask(predicted, min_area=100, include_masks=False)

            return jsonify({
                'success': True,
                'num_classes': int(predicted.max() + 1),
                'width': int(predicted.shape[1]),
                'height': int(predicted.shape[0]),
                'bounding_boxes': bounding_boxes,
            })

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            inference_stats['active_requests'] -= 1


@app.route('/api/gnu-mapping/pipe-presets', methods=['GET'])
def gnu_mapping_pipe_presets():
    """관종별 직경 프리셋 (현장 관경 + 논문 시편)"""
    from gnu_mapping import PIPE_DIMENSIONS, PIPE_GROUP_LABELS
    return jsonify({
        'success': True,
        'presets': PIPE_DIMENSIONS,
        'group_labels': PIPE_GROUP_LABELS,
    })


@app.route('/api/gnu-mapping/evaluate/sample', methods=['POST'])
def gnu_mapping_evaluate_sample():
    """단일 샘플 상세 결과 — 원본/전개도/오버레이 + 메트릭

    Body:
      img_path, mask_path: 샘플 파일 경로 (evaluate 응답에서 받은 값)
      pipe_type or pipe_diameter_mm, water, pixel_per_mm, max_depth_mm
    """
    try:
        from gnu_mapping import PerformanceEvaluator, PIPE_DIMENSIONS

        data = request.json or {}
        img_path = data.get('img_path')
        mask_path = data.get('mask_path')
        if not img_path or not os.path.exists(img_path):
            return jsonify({'success': False, 'error': 'img_path not found'}), 400
        if not mask_path or not os.path.exists(mask_path):
            return jsonify({'success': False, 'error': 'mask_path not found'}), 400

        pipe_type = data.get('pipe_type')
        pipe_diameter = data.get('pipe_diameter_mm')
        water = data.get('water')
        pixel_per_mm = data.get('pixel_per_mm', 10.0)
        max_depth_mm = data.get('max_depth_mm', 300)

        if pipe_type and pipe_type in PIPE_DIMENSIONS:
            preset = PIPE_DIMENSIONS[pipe_type]
            if pipe_diameter is None:
                pipe_diameter = preset['diameter_mm']
            if water is None:
                water = preset['water_default']
        if pipe_diameter is None:
            return jsonify({'success': False,
                            'error': 'pipe_type or pipe_diameter_mm required'}), 400

        ppnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'weights', 'ppnet.pt')
        if not os.path.exists(ppnet_path):
            return jsonify({'success': False, 'error': 'PPNet model not found'}), 400

        evaluator = PerformanceEvaluator(
            ppnet_model_path=ppnet_path,
            pipe_diameter_mm=pipe_diameter,
            water=bool(water),
            pixel_per_mm=pixel_per_mm,
            max_depth_mm=max_depth_mm,
        )
        detail = evaluator.evaluate_sample(img_path, mask_path)
        return jsonify({'success': True, 'pipe_type': pipe_type, **detail})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/gnu-mapping/evaluate', methods=['POST'])
def gnu_mapping_evaluate():
    """MAPE 평가 — Reference Marker(19mm) 매핑 정확도

    Body:
      directory: 이미지/마스크 페어가 있는 절대 경로 (필수)
      pipe_type: CIP|PVC|CP|PP (선택, diameter/water 자동 설정)
      pipe_diameter_mm, water, pixel_per_mm, max_depth_mm (선택)
    """
    try:
        from gnu_mapping import PerformanceEvaluator, PIPE_DIMENSIONS

        data = request.json or {}
        directory = data.get('directory')
        if not directory or not os.path.isdir(directory):
            return jsonify({'success': False, 'error': 'valid directory required'}), 400

        pipe_type = data.get('pipe_type')
        pipe_diameter = data.get('pipe_diameter_mm')
        water = data.get('water')
        pixel_per_mm = data.get('pixel_per_mm', 10.0)
        max_depth_mm = data.get('max_depth_mm', 300)

        if pipe_type and pipe_type in PIPE_DIMENSIONS:
            preset = PIPE_DIMENSIONS[pipe_type]
            if pipe_diameter is None:
                pipe_diameter = preset['diameter_mm']
            if water is None:
                water = preset['water_default']

        if pipe_diameter is None:
            return jsonify({'success': False,
                            'error': 'pipe_type or pipe_diameter_mm required'}), 400

        ppnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'weights', 'ppnet.pt')
        if not os.path.exists(ppnet_path):
            return jsonify({'success': False, 'error': 'PPNet model not found'}), 400

        evaluator = PerformanceEvaluator(
            ppnet_model_path=ppnet_path,
            pipe_diameter_mm=pipe_diameter,
            water=bool(water),
            pixel_per_mm=pixel_per_mm,
            max_depth_mm=max_depth_mm,
        )
        result = evaluator.evaluate(directory)
        return jsonify({
            'success': True,
            'pipe_type': pipe_type,
            **result,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/gnu-mapping/process', methods=['POST'])
def gnu_mapping_process():
    """GNU Mapping — 영상에서 프레임 추출 + PPNet + 3D→2D 전개도"""
    try:
        from gnu_mapping import GNUMappingEngine
        import math

        data = request.json

        # 입력: video_path + frame_number 또는 image_base64
        video_path = data.get('video_path')
        frame_number = data.get('frame_number', 0)
        img_b64 = data.get('image_base64')

        pipe_type = data.get('pipe_type')
        pipe_diameter = data.get('pipe_diameter_mm', 80)
        water = data.get('water', False)
        pixel_per_mm = data.get('pixel_per_mm', 10.0)
        max_depth_mm = data.get('max_depth_mm', 300)

        # pipe_type 선택 시 프리셋 우선 적용
        from gnu_mapping import PIPE_DIMENSIONS as _PRESETS
        if pipe_type and pipe_type in _PRESETS:
            preset = _PRESETS[pipe_type]
            pipe_diameter = preset['diameter_mm']
            if 'water' not in data:
                water = preset['water_default']

        # 프레임 로드
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame_bgr = cap.read()
            cap.release()
            if not ret:
                return jsonify({'success': False, 'error': f'Failed to read frame {frame_number}'}), 400
        elif img_b64:
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return jsonify({'success': False, 'error': 'Failed to decode image'}), 400
        else:
            return jsonify({'success': False, 'error': 'video_path or image_base64 required'}), 400

        h, w = frame_bgr.shape[:2]

        # PPNet 모델 경로
        ppnet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'ppnet.pt')
        if not os.path.exists(ppnet_path):
            return jsonify({'success': False, 'error': 'PPNet model not found'}), 400

        # ── 어노테이션 → 결함 마스크 변환 ──
        defect_masks = []
        annotations = data.get('annotations', [])
        if annotations:
            for ann in annotations:
                polygon = ann.get('polygon', [])
                label = ann.get('label', ann.get('category', 'defect'))
                if not polygon or len(polygon) < 3:
                    continue
                mask = np.zeros((h, w), dtype=np.uint8)
                pts = []
                for p in polygon:
                    if isinstance(p, dict):
                        pts.append([int(p.get('x', 0)), int(p.get('y', 0))])
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        pts.append([int(p[0]), int(p[1])])
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
                    defect_masks.append({'label': label, 'mask': mask})

        # ── GNU Mapping ──
        engine = GNUMappingEngine(
            ppnet_model_path=ppnet_path,
            pipe_diameter_mm=pipe_diameter,
            water=water,
            pixel_per_mm=pixel_per_mm,
            max_depth_mm=max_depth_mm,
        )
        include_depth_map = bool(data.get('include_depth_map', False))
        gnu_result = engine.process_frame(frame_bgr, defect_masks or None,
                                           include_depth_map=include_depth_map)

        # 원본 프레임 (base64)
        _, fbuf = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(fbuf).decode('utf-8')

        return jsonify({
            'success': True,
            **gnu_result,
            'frame_b64': frame_b64,
            'frame_width': w,
            'frame_height': h,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/survey/infer', methods=['POST'])
def survey_yolo_infer():
    """Survey용 YOLO instance segmentation 추론 — base64 이미지 입력"""
    global yolo_model, yolo_initialized

    with inference_lock:
        inference_stats['total_requests'] += 1
        inference_stats['active_requests'] += 1
        try:
            if not yolo_initialized or yolo_model is None:
                # YOLO가 아직 초기화 안 된 경우 자동 로드 시도
                if not load_yolo_model():
                    return jsonify({'success': False, 'error': 'YOLO model not initialized'}), 400

            data = request.json
            img_b64 = data.get('image_base64')
            if not img_b64:
                return jsonify({'success': False, 'error': 'image_base64 required'}), 400

            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

            height, width = frame.shape[:2]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # YOLO 추론
            yolo_results = yolo_model(frame_rgb, verbose=False, stream=True)
            result = next(yolo_results)

            from pipe_survey import yolo_result_to_detections
            detections = yolo_result_to_detections(result, width, height)

            return jsonify({
                'success': True,
                'width': width,
                'height': height,
                'detections': detections,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            inference_stats['active_requests'] -= 1


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
    """추론 작업의 최신 처리된 프레임 이미지 반환 (메모리에서)"""
    with job_lock:
        if job_id not in active_jobs:
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

        job = active_jobs[job_id]
        latest_frame = job.get('latest_frame')

        if not latest_frame:
            return jsonify({
                'success': False,
                'error': 'No preview frame available yet'
            }), 404

        # 메모리에서 바로 반환
        frame_data = latest_frame

    # 이미지 데이터 반환
    try:
        from io import BytesIO
        return send_file(BytesIO(frame_data), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/inference/frames/<job_id>', methods=['GET'])
def get_inference_frames(job_id):
    """추론 작업의 처리된 프레임 정보 반환"""
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
            'total_frames': job.get('total_frames', 0),
            'processed_frames': job.get('current_frame', 0),
            'fps': job.get('fps', 30),
            'current_frame': job.get('current_frame', 0)
        })


@app.route('/api/inference/frame/<job_id>/<int:frame_index>', methods=['GET'])
def get_inference_frame(job_id, frame_index):
    """특정 프레임 이미지 반환 - 현재 미지원 (JSON 결과만 저장)"""
    # 프레임 이미지는 더 이상 저장하지 않음
    # 필요 시 원본 비디오에서 추출하여 사용
    return jsonify({
        'success': False,
        'error': 'Individual frame images are not stored. Use inference_results.json and original video.'
    }), 404


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

                # results 키에서 프레임 수 확인 (JSON 기반 결과)
                results = result_data.get('results', {})
                frame_count = len(results) if isinstance(results, dict) else result_data.get('total_frames', 0)

                return jsonify({
                    'success': True,
                    'exists': True,
                    'result_path': output_path,
                    'total_frames': result_data.get('total_frames', frame_count),
                    'fps': result_data.get('fps', 30),
                    'video_path': result_data.get('video_path', video_path),
                    'frame_count': frame_count,
                    'width': result_data.get('width'),
                    'height': result_data.get('height'),
                    'model_type': result_data.get('model_type')
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
    """완료된 추론 결과 프레임 이미지 반환 (파일 경로 기반) - deprecated"""
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


@app.route('/api/inference/results', methods=['POST'])
def get_inference_results():
    """추론 결과 JSON 반환 (클라이언트 렌더링용)"""
    try:
        data = request.get_json()
        result_path = data.get('result_path')

        if not result_path:
            return jsonify({'success': False, 'error': 'Result path required'}), 400

        # 상대 경로를 절대 경로로 변환
        if not os.path.isabs(result_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result_path = os.path.join(base_dir, result_path)

        result_json_path = os.path.join(result_path, 'inference_results.json')

        if not os.path.exists(result_json_path):
            return jsonify({'success': False, 'error': 'Result file not found'}), 404

        # JSON 파일 반환
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)

        return jsonify({
            'success': True,
            'data': result_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/inference/analyze-motion', methods=['POST'])
def analyze_motion():
    """비디오 움직임 분석 및 정지 구간 탐지 (스트리밍 진행 상황 지원)"""
    data = request.get_json()
    result_path = data.get('result_path')
    motion_threshold = data.get('motion_threshold', 5.0)
    min_segment_duration = data.get('min_segment_duration', 1.0)
    stream_progress = data.get('stream_progress', False)

    if not result_path:
        return jsonify({'success': False, 'error': 'Result path required'}), 400

    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(result_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_path = os.path.join(base_dir, result_path)

    result_json_path = os.path.join(result_path, 'inference_results.json')

    if not os.path.exists(result_json_path):
        return jsonify({'success': False, 'error': 'Result file not found'}), 404

    def generate():
        try:
            # 추론 결과 로드
            with open(result_json_path, 'r') as f:
                result_data = json.load(f)

            video_path = result_data.get('video_path')
            fps = result_data.get('fps', 25)
            results = result_data.get('results', [])

            if not video_path or not os.path.exists(video_path):
                yield f"data: {json.dumps({'success': False, 'error': f'Video not found: {video_path}'})}\n\n"
                return

            print(f"[MOTION] Analyzing video: {video_path}", flush=True)

            # 비디오 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                yield f"data: {json.dumps({'success': False, 'error': 'Failed to open video'})}\n\n"
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = min(total_frames, len(results)) if results else total_frames

            # 프레임 간 움직임 계산 (5프레임마다 샘플링)
            sample_interval = 5
            motion_scores = []
            prev_gray = None
            last_progress = -1

            for frame_idx in range(0, max_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 180))

                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    motion = float(diff.mean())
                    motion_scores.append({'frame': frame_idx, 'motion': motion})

                prev_gray = gray

                # 진행 상황 전송 (5% 단위)
                progress = int((frame_idx / max_frames) * 100)
                if progress >= last_progress + 5:
                    last_progress = progress
                    yield f"data: {json.dumps({'progress': progress, 'current_frame': frame_idx, 'total_frames': max_frames})}\n\n"

            cap.release()

            # 정지 구간 탐지
            segments = []
            segment_start = None

            for item in motion_scores:
                frame_idx = item['frame']
                motion = item['motion']

                if motion < motion_threshold:
                    if segment_start is None:
                        segment_start = frame_idx
                else:
                    if segment_start is not None:
                        segment_end = frame_idx
                        duration = (segment_end - segment_start) / fps
                        if duration >= min_segment_duration:
                            segments.append({
                                'start': segment_start,
                                'end': segment_end,
                                'duration': round(duration, 1),
                                'start_time': round(segment_start / fps, 1),
                                'end_time': round(segment_end / fps, 1)
                            })
                        segment_start = None

            # 마지막 구간 처리
            if segment_start is not None:
                segment_end = motion_scores[-1]['frame'] if motion_scores else total_frames
                duration = (segment_end - segment_start) / fps
                if duration >= min_segment_duration:
                    segments.append({
                        'start': segment_start,
                        'end': segment_end,
                        'duration': round(duration, 1),
                        'start_time': round(segment_start / fps, 1),
                        'end_time': round(segment_end / fps, 1)
                    })

            print(f"[MOTION] Found {len(segments)} stationary segments", flush=True)

            # 최종 결과 전송
            yield f"data: {json.dumps({'success': True, 'total_frames': total_frames, 'fps': fps, 'motion_threshold': motion_threshold, 'min_segment_duration': min_segment_duration, 'segments': segments, 'segment_count': len(segments), 'motion_stats': {'min': round(min(m['motion'] for m in motion_scores), 2) if motion_scores else 0, 'max': round(max(m['motion'] for m in motion_scores), 2) if motion_scores else 0, 'avg': round(sum(m['motion'] for m in motion_scores) / len(motion_scores), 2) if motion_scores else 0}})}\n\n"

        except Exception as e:
            print(f"[MOTION] Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'success': False, 'error': str(e)})}\n\n"

    if stream_progress:
        return Response(generate(), mimetype='text/event-stream')
    else:
        # 기존 방식 (스트리밍 없이)
        try:
            with open(result_json_path, 'r') as f:
                result_data = json.load(f)

            video_path = result_data.get('video_path')
            fps = result_data.get('fps', 25)
            results = result_data.get('results', [])

            if not video_path or not os.path.exists(video_path):
                return jsonify({'success': False, 'error': f'Video not found: {video_path}'}), 404

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return jsonify({'success': False, 'error': 'Failed to open video'}), 500

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = min(total_frames, len(results)) if results else total_frames
            sample_interval = 5
            motion_scores = []
            prev_gray = None

            for frame_idx in range(0, max_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 180))
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    motion = float(diff.mean())
                    motion_scores.append({'frame': frame_idx, 'motion': motion})
                prev_gray = gray

            cap.release()

            segments = []
            segment_start = None
            for item in motion_scores:
                frame_idx = item['frame']
                motion = item['motion']
                if motion < motion_threshold:
                    if segment_start is None:
                        segment_start = frame_idx
                else:
                    if segment_start is not None:
                        segment_end = frame_idx
                        duration = (segment_end - segment_start) / fps
                        if duration >= min_segment_duration:
                            segments.append({'start': segment_start, 'end': segment_end, 'duration': round(duration, 1), 'start_time': round(segment_start / fps, 1), 'end_time': round(segment_end / fps, 1)})
                        segment_start = None

            if segment_start is not None:
                segment_end = motion_scores[-1]['frame'] if motion_scores else total_frames
                duration = (segment_end - segment_start) / fps
                if duration >= min_segment_duration:
                    segments.append({'start': segment_start, 'end': segment_end, 'duration': round(duration, 1), 'start_time': round(segment_start / fps, 1), 'end_time': round(segment_end / fps, 1)})

            return jsonify({'success': True, 'total_frames': total_frames, 'fps': fps, 'motion_threshold': motion_threshold, 'min_segment_duration': min_segment_duration, 'segments': segments, 'segment_count': len(segments), 'motion_stats': {'min': round(min(m['motion'] for m in motion_scores), 2) if motion_scores else 0, 'max': round(max(m['motion'] for m in motion_scores), 2) if motion_scores else 0, 'avg': round(sum(m['motion'] for m in motion_scores) / len(motion_scores), 2) if motion_scores else 0}})

        except Exception as e:
            print(f"[MOTION] Error: {e}", flush=True)
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/inference/extract-representatives', methods=['POST'])
def extract_representative_frames():
    """정지 구간에서 대표 프레임 추출"""
    try:
        data = request.get_json()
        result_path = data.get('result_path')
        segments = data.get('segments', [])
        frames_per_segment = data.get('frames_per_segment', 3)
        min_confidence = data.get('min_confidence', 0.5)

        if not result_path:
            return jsonify({'success': False, 'error': 'Result path required'}), 400

        if not segments:
            return jsonify({'success': False, 'error': 'No segments provided'}), 400

        # 상대 경로를 절대 경로로 변환
        if not os.path.isabs(result_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result_path = os.path.join(base_dir, result_path)

        result_json_path = os.path.join(result_path, 'inference_results.json')

        if not os.path.exists(result_json_path):
            return jsonify({'success': False, 'error': 'Result file not found'}), 404

        # 추론 결과 로드
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)

        fps = result_data.get('fps', 25)
        results = result_data.get('results', [])

        # 프레임별 검출 수 계산 (confidence 필터링 적용)
        frame_detection_counts = {}
        frame_detections = {}
        for item in results:
            frame_num = item['frame_number']
            detections = [d for d in item.get('detections', []) if d.get('confidence', 0) >= min_confidence]
            frame_detection_counts[frame_num] = len(detections)
            frame_detections[frame_num] = detections

        # 각 구간에서 검출이 가장 많은 프레임 선택
        representative_frames = []

        for seg_idx, segment in enumerate(segments):
            start = segment['start']
            end = segment['end']

            # 구간 내 프레임들의 검출 수 확인
            segment_frames = []
            for frame_num in range(start, end + 1):
                if frame_num in frame_detection_counts:
                    segment_frames.append((frame_num, frame_detection_counts[frame_num]))

            # 검출 수 기준 상위 N개 선택
            segment_frames.sort(key=lambda x: -x[1])
            top_frames = segment_frames[:frames_per_segment]

            for frame_num, det_count in top_frames:
                if det_count > 0:  # 검출이 있는 프레임만
                    representative_frames.append({
                        'segment_index': seg_idx,
                        'frame_number': frame_num,
                        'time': round(frame_num / fps, 2),
                        'detection_count': det_count,
                        'detections': frame_detections.get(frame_num, [])
                    })

        # 프레임 번호 순으로 정렬
        representative_frames.sort(key=lambda x: x['frame_number'])

        total_detections = sum(f['detection_count'] for f in representative_frames)

        print(f"[EXTRACT] Selected {len(representative_frames)} representative frames with {total_detections} detections", flush=True)

        return jsonify({
            'success': True,
            'frames_per_segment': frames_per_segment,
            'min_confidence': min_confidence,
            'total_frames': len(representative_frames),
            'total_detections': total_detections,
            'representative_frames': representative_frames
        })

    except Exception as e:
        print(f"[EXTRACT] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/inference/export-dataset', methods=['POST'])
def export_dataset_from_inference():
    """대표 프레임을 데이터셋으로 내보내기"""
    try:
        data = request.get_json()
        result_path = data.get('result_path')
        representative_frames = data.get('representative_frames', [])
        output_dir = data.get('output_dir', 'extracted_dataset')
        format_type = data.get('format', 'yolo')  # 'yolo', 'coco', 'annotation'
        split_ratio = data.get('split_ratio', [0.8, 0.1, 0.1])  # train, val, test

        if not result_path:
            return jsonify({'success': False, 'error': 'Result path required'}), 400

        if not representative_frames:
            return jsonify({'success': False, 'error': 'No frames provided'}), 400

        # 상대 경로를 절대 경로로 변환
        if not os.path.isabs(result_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            result_path = os.path.join(base_dir, result_path)

        result_json_path = os.path.join(result_path, 'inference_results.json')

        if not os.path.exists(result_json_path):
            return jsonify({'success': False, 'error': 'Result file not found'}), 404

        # 추론 결과 로드
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)

        video_path = result_data.get('video_path')
        width = result_data.get('width', 1920)
        height = result_data.get('height', 1080)

        if not video_path or not os.path.exists(video_path):
            return jsonify({'success': False, 'error': f'Video not found: {video_path}'}), 404

        # 출력 디렉토리 생성
        if not os.path.isabs(output_dir):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, output_dir)

        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"{output_dir}_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Failed to open video'}), 500

        # 클래스 목록 수집
        class_names = set()
        for frame_data in representative_frames:
            for det in frame_data.get('detections', []):
                class_names.add(det.get('label', 'unknown'))
        class_names = sorted(list(class_names))
        class_to_id = {name: idx for idx, name in enumerate(class_names)}

        print(f"[EXPORT] Classes: {class_names}", flush=True)

        # 데이터 분할
        import random
        random.shuffle(representative_frames)

        n_total = len(representative_frames)
        n_train = int(n_total * split_ratio[0])
        n_val = int(n_total * split_ratio[1])

        splits = {
            'train': representative_frames[:n_train],
            'val': representative_frames[n_train:n_train + n_val],
            'test': representative_frames[n_train + n_val:]
        }

        stats = {'train': 0, 'val': 0, 'test': 0, 'total_images': 0, 'total_annotations': 0}

        if format_type == 'yolo':
            # YOLO 형식으로 내보내기
            for split_name, frames in splits.items():
                img_dir = os.path.join(output_dir, split_name, 'images')
                lbl_dir = os.path.join(output_dir, split_name, 'labels')
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(lbl_dir, exist_ok=True)

                for frame_data in frames:
                    frame_num = frame_data['frame_number']
                    detections = frame_data.get('detections', [])

                    # 프레임 추출
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # 이미지 저장
                    img_filename = f"frame_{frame_num:06d}.jpg"
                    img_path = os.path.join(img_dir, img_filename)
                    cv2.imwrite(img_path, frame)

                    # 라벨 파일 생성 (YOLO 세그멘테이션 형식)
                    lbl_filename = f"frame_{frame_num:06d}.txt"
                    lbl_path = os.path.join(lbl_dir, lbl_filename)

                    with open(lbl_path, 'w') as f:
                        for det in detections:
                            class_id = class_to_id.get(det.get('label', 'unknown'), 0)
                            polygon = det.get('polygon', [])

                            if polygon:
                                # YOLO 세그멘테이션: class_id x1 y1 x2 y2 ... (normalized)
                                coords = []
                                for point in polygon:
                                    x_norm = point[0] / width
                                    y_norm = point[1] / height
                                    coords.extend([x_norm, y_norm])

                                coord_str = ' '.join(f"{c:.6f}" for c in coords)
                                f.write(f"{class_id} {coord_str}\n")
                                stats['total_annotations'] += 1

                    stats[split_name] += 1
                    stats['total_images'] += 1

            # data.yaml 생성
            yaml_content = f"""# Dataset exported from inference results
# Generated: {timestamp}

path: {output_dir}
train: train/images
val: val/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
            with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
                f.write(yaml_content)

        cap.release()

        print(f"[EXPORT] Completed: {stats}", flush=True)

        return jsonify({
            'success': True,
            'output_dir': output_dir,
            'format': format_type,
            'classes': class_names,
            'stats': stats
        })

    except Exception as e:
        print(f"[EXPORT] Error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def process_video_inference(job_id, video_path, output_path, model_type):
    """백그라운드에서 비디오 추론 실행 (SegFormer 또는 YOLO)"""
    global segformer_model, segformer_processor, segformer_device, yolo_model

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
            active_jobs[job_id]['latest_frame'] = None  # 최신 프레임 (메모리, 미리보기용)

        # 결과 저장용
        results = []
        frame_count = 0
        preview_interval = 10  # 미리보기는 10프레임마다 업데이트

        # 클래스별 색상 정의 (YOLO용)
        yolo_colors = [
            (255, 0, 0),      # 빨강
            (255, 255, 0),    # 노랑
            (0, 255, 0),      # 초록
            (0, 255, 255),    # 청록
            (255, 0, 255),    # 마젠타
            (128, 0, 255),    # 보라
            (255, 128, 0),    # 주황
            (0, 128, 255),    # 하늘
        ]

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

            # 미리보기 생성 여부 (N프레임마다)
            should_generate_preview = (frame_count % preview_interval == 0)

            if model_type == 'yolo':
                # YOLO 추론 (stream=True로 메모리 효율화)
                yolo_results = yolo_model(frame_rgb, verbose=False, stream=True)
                result = next(yolo_results)

                frame_detections = []

                # 탐지 결과 처리
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    confs = result.boxes.conf.cpu().numpy()

                    # 세그멘테이션 마스크가 있는 경우
                    masks = result.masks.data.cpu().numpy() if result.masks is not None else None

                    # 미리보기용 오버레이 (필요할 때만 생성)
                    if should_generate_preview:
                        overlay = frame.copy()

                    for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
                        x1, y1, x2, y2 = box.astype(int)
                        class_name = result.names[cls] if cls < len(result.names) else f'class_{cls}'
                        color = yolo_colors[cls % len(yolo_colors)]

                        detection = {
                            'box': [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # [x, y, w, h]
                            'label': class_name,
                            'class_id': int(cls),
                            'confidence': round(float(conf), 3)  # 소수점 3자리로 제한
                        }

                        # 세그멘테이션 마스크가 있으면 폴리곤 추출 (CPU 부하가 크므로 매 프레임 처리)
                        # 단, 마스크 리사이즈와 컨투어 추출은 최적화
                        if masks is not None and i < len(masks):
                            mask = masks[i]
                            # 마스크를 원본 크기로 리사이즈 (더 작은 크기로 먼저 처리)
                            mask_h, mask_w = mask.shape
                            # 작은 스케일로 컨투어 추출 후 스케일업
                            scale = 0.25  # 1/4 크기로 처리
                            small_w, small_h = int(width * scale), int(height * scale)
                            mask_small = cv2.resize(mask, (small_w, small_h))
                            mask_binary = (mask_small > 0.5).astype(np.uint8) * 255

                            # 컨투어 추출
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                # 가장 큰 컨투어 선택
                                largest_contour = max(contours, key=cv2.contourArea)
                                # 폴리곤 단순화 (세밀하게 - 인스턴스 세그멘테이션 학습용)
                                epsilon = 0.003 * cv2.arcLength(largest_contour, True)  # 0.3% 단순화
                                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                                # 원본 크기로 스케일업, 최대 150포인트로 제한
                                polygon_points = (approx.reshape(-1, 2) / scale).astype(int)
                                if len(polygon_points) > 150:
                                    # 균등하게 샘플링
                                    indices = np.linspace(0, len(polygon_points) - 1, 150, dtype=int)
                                    polygon_points = polygon_points[indices]
                                detection['polygon'] = polygon_points.tolist()

                                # 미리보기용 오버레이 (필요할 때만)
                                if should_generate_preview:
                                    approx_scaled = (approx.reshape(-1, 2) / scale).astype(int)
                                    mask_overlay = np.zeros_like(overlay)
                                    cv2.fillPoly(mask_overlay, [approx_scaled], color)
                                    overlay = cv2.addWeighted(overlay, 1.0, mask_overlay, 0.3, 0)

                        frame_detections.append(detection)

                        # 미리보기용 바운딩 박스 (필요할 때만)
                        if should_generate_preview:
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                            label_text = f'{class_name} {conf:.2f}'
                            cv2.putText(overlay, label_text, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                frame_results = {
                    'frame_number': frame_count,
                    'detections': frame_detections
                }

            else:
                # SegFormer 추론 (기존 로직)
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

                # 미리보기용 오버레이 (필요할 때만)
                if should_generate_preview:
                    mask_colored = np.zeros((height, width, 3), dtype=np.uint8)
                    mask_colored[pred_seg == 1] = [255, 0, 0]  # rust
                    mask_colored[pred_seg == 2] = [255, 255, 0]  # scale
                    overlay = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

            results.append(frame_results)
            frame_count += 1

            # 진행 상황 업데이트
            progress = (frame_count / total_frames) * 100

            # 미리보기 생성 (N프레임마다만)
            if should_generate_preview:
                _, encoded_frame = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 70])
                with job_lock:
                    active_jobs[job_id]['current_frame'] = frame_count
                    active_jobs[job_id]['progress'] = progress
                    active_jobs[job_id]['latest_frame'] = encoded_frame.tobytes()
            else:
                # 프레임 카운트만 업데이트 (lock 최소화)
                with job_lock:
                    active_jobs[job_id]['current_frame'] = frame_count
                    active_jobs[job_id]['progress'] = progress

        cap.release()

        # 결과 JSON 저장 (indent 없이 저장하여 파일 크기 최소화)
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
            }, f, separators=(',', ':'))  # 공백 없이 저장

        # 작업 완료 상태 업데이트
        with job_lock:
            active_jobs[job_id]['status'] = 'completed'
            active_jobs[job_id]['progress'] = 100
            active_jobs[job_id]['result_file'] = result_json_path

            # 임시 비디오 파일 삭제
            temp_video = active_jobs[job_id].get('temp_video_path')
            if temp_video and os.path.exists(temp_video):
                try:
                    os.remove(temp_video)
                    print(f"[INFERENCE] Temp video deleted: {temp_video}", flush=True)
                except Exception as del_err:
                    print(f"[INFERENCE] Failed to delete temp video: {del_err}", flush=True)

    except Exception as e:
        import traceback
        traceback.print_exc()
        with job_lock:
            if job_id in active_jobs:
                active_jobs[job_id]['status'] = 'failed'
                active_jobs[job_id]['error'] = str(e)

                # 실패 시에도 임시 파일 삭제
                temp_video = active_jobs[job_id].get('temp_video_path')
                if temp_video and os.path.exists(temp_video):
                    try:
                        os.remove(temp_video)
                    except:
                        pass


@app.route('/api/inference', methods=['POST'])
def run_video_inference():
    """전체 비디오에 대한 추론 실행 (비동기)"""
    global segformer_model, segformer_processor, segformer_device, ai_initialized
    global yolo_model, yolo_initialized

    try:
        data = request.json
        print(f"[DEBUG] Inference request data: {data}", flush=True)

        model_type = data.get('model_type', 'segformer')
        model_path = data.get('model_path')

        # 모델 타입에 따른 초기화 확인
        if model_type == 'yolo':
            if not yolo_initialized or yolo_model is None:
                # YOLO 모델 자동 초기화 시도
                print("[DEBUG] YOLO model not initialized, attempting to load...")
                if not load_yolo_model(model_path):
                    return jsonify({
                        'success': False,
                        'error': 'YOLO model not initialized. Call /api/ai/initialize/yolo first.'
                    }), 400
        else:
            if not ai_initialized or segformer_model is None:
                return jsonify({
                    'success': False,
                    'error': 'AI model not initialized. Call /api/ai/initialize first.'
                }), 400
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
        data = request.get_json(force=True, silent=True)
        if data is None:
            print(f"[DATASET BUILD] ERROR: Failed to parse JSON. Content-Length: {request.content_length}")
            return jsonify({'success': False, 'error': 'Failed to parse JSON request'}), 400
        print(f"[DATASET BUILD] Received data keys: {data.keys() if data else 'None'}")
        annotations_data = data.get('annotations_data', [])
        print(f"[DATASET BUILD] Received keys: {list(data.keys())}")
        print(f"[DATASET BUILD] annotations_data length: {len(annotations_data)}")
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
            user_id = anno_data.get('user_id', 'unknown')
            project_id = anno_data.get('project_id', '')
            video_id = anno_data.get('video_id', '')
            annotations = anno_data.get('annotations', {})
            project_dir = Path(anno_data.get('project_dir', ''))

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
        frame_cache_root = Path(__file__).parent / 'frame_cache'

        def get_cache_path(video_path, frame_num):
            key = hashlib.sha1(video_path.encode('utf-8')).hexdigest()
            return frame_cache_root / key[:2] / key / f"{int(frame_num):06d}.jpg"

        def process_frames(frames, split_name):
            saved_count = 0
            cache_hits = 0
            cache_misses = 0

            frames_by_video = defaultdict(list)
            for frame_data in frames:
                frames_by_video[frame_data['video_path']].append(frame_data)

            def process_video(video_path, video_frames):
                local_count = 0
                local_hits = 0
                local_misses = 0
                cap = None

                try:
                    video_frames.sort(key=lambda x: x['frame_num'])

                    for frame_data in video_frames:
                        fnum = frame_data['frame_num']
                        image_filename = f"{frame_data['project_id']}_{frame_data['video_id']}_frame{fnum}.jpg"
                        image_path = output_path / split_name / 'images' / image_filename
                        label_path = output_path / split_name / 'labels' / image_filename.replace('.jpg', '.txt')

                        cache_path = get_cache_path(video_path, fnum)
                        frame = None

                        # 1) 캐시 우선 사용
                        if cache_path.exists():
                            frame = cv2.imread(str(cache_path))
                            if frame is not None:
                                local_hits += 1
                                try:
                                    os.link(str(cache_path), str(image_path))
                                except Exception:
                                    shutil.copy2(str(cache_path), str(image_path))

                        # 2) 캐시 미스면 비디오에서 추출 + 캐시에 저장
                        if frame is None:
                            local_misses += 1

                            if cap is None:
                                cap = cv2.VideoCapture(video_path)
                                if not cap.isOpened():
                                    print(f"[DATASET BUILD-FILTERED] Cannot open video: {video_path}")
                                    break

                            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                continue

                            cv2.imwrite(str(image_path), frame)

                            try:
                                cache_path.parent.mkdir(parents=True, exist_ok=True)
                                if not cache_path.exists():
                                    cv2.imwrite(str(cache_path), frame)
                            except Exception as e:
                                print(f"[DATASET BUILD-FILTERED] cache write failed: {cache_path} ({e})")

                        h, w = frame.shape[:2]
                        lines = []
                        for anno in frame_data['annotations']:
                            cid = class_to_id.get(anno['label'])
                            if cid is None:
                                continue
                            coords = []
                            for point in anno['polygon']:
                                try:
                                    x = float(point['x']) / w
                                    y = float(point['y']) / h
                                except Exception:
                                    x = float(point[0]) / w
                                    y = float(point[1]) / h
                                x = max(0.0, min(1.0, x))
                                y = max(0.0, min(1.0, y))
                                coords.append(f"{x:.6f} {y:.6f}")
                            if len(coords) >= 3:
                                lines.append(f"{cid} " + ' '.join(coords))

                        if lines:
                            with open(label_path, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(lines))
                            local_count += 1
                        else:
                            try:
                                image_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                finally:
                    if cap is not None:
                        cap.release()

                return local_count, local_hits, local_misses

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_video, vp, vf) for vp, vf in frames_by_video.items()]
                for fut in as_completed(futures):
                    try:
                        c, h, m = fut.result()
                        saved_count += c
                        cache_hits += h
                        cache_misses += m
                    except Exception as e:
                        print(f"[DATASET BUILD-FILTERED] Video worker error: {e}")

            print(f"[DATASET BUILD-FILTERED] {split_name}: saved={saved_count}, cache_hit={cache_hits}, cache_miss={cache_misses}")
            return saved_count, cache_hits, cache_misses

        train_count, train_hits, train_misses = process_frames(train_frames, 'train')
        val_count, val_hits, val_misses = process_frames(val_frames, 'val')
        test_count, test_hits, test_misses = process_frames(test_frames, 'test')
        val_count = process_frames(val_frames, 'val')
        test_count = process_frames(test_frames, 'test')

        print(f"[DATASET BUILD] Saved - Train: {train_count}, Val: {val_count}, Test: {test_count}")

        # 한글 클래스 이름을 영문으로 매핑 (YOLO 호환성)
        korean_to_english = {
            '정상부': 'normal',
            '변형': 'deformation',
            '균열': 'crack',
            '부식': 'corrosion',
            '침전물(흙)': 'sediment_soil',
            '침전물(모래)': 'sediment_sand',
            '침전물(부식 생성물)': 'sediment_corrosion',
            '침전물(탈리, 도장재)': 'sediment_coating',
            '침전물(기타)': 'sediment_other',
            '슬라임(물때)': 'slime',
            '논의필요': 'needs_discussion',
            '소실점': 'vanishing_point'
        }

        # 실제 어노테이션에서 사용된 label 필드 수집
        # (project.json 인덱스 기반 매핑보다 실제 label 필드가 우선)
        class_id_to_label = {}
        for frame_data in all_frames:
            for anno in frame_data['annotations']:
                class_id = anno.get('class_id', 0)
                label = anno.get('label')
                if label:
                    # 한글이면 영문으로 변환, 이미 영문이면 그대로 사용
                    english_label = korean_to_english.get(label, label)
                    class_id_to_label[class_id] = english_label

        # 모든 프로젝트의 클래스 매핑을 병합 (fallback용)
        merged_class_mapping = {}
        for project_dir_str, class_mapping in project_classes.items():
            merged_class_mapping.update(class_mapping)

        # 사용된 클래스 정보 정리 (label 필드 우선, 없으면 project.json 사용)
        sorted_class_ids = sorted(used_classes)
        class_names_list = []
        for cid in sorted_class_ids:
            if cid in class_id_to_label:
                class_names_list.append(class_id_to_label[cid])
            else:
                # project.json fallback도 한글이면 영문으로 변환
                fallback_name = merged_class_mapping.get(cid, f'class_{cid}')
                english_name = korean_to_english.get(fallback_name, fallback_name)
                class_names_list.append(english_name)
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




@app.route('/api/dataset/build_yolo_filtered', methods=['POST'])
def build_yolo_dataset_filtered():
    """다중 프로젝트 YOLO 데이터셋 빌드 (클래스 필터 + class id 재매핑)"""
    from pathlib import Path
    import random
    import hashlib
    import shutil
    from datetime import datetime
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        data = request.get_json(force=True, silent=True) or {}
        annotations_data = data.get('annotations_data', [])
        projects = data.get('projects', {}) or {}
        base_projects_dir = Path(data.get('base_projects_dir', '/home/intu/Nas2/k_water/pipe_inspector_data'))
        selected_classes = data.get('classes', []) or []
        output_dir = data.get('output_dir', 'pipe_dataset')
        split_ratio = data.get('split_ratio', '0.7,0.15,0.15')

        # Compact request mode: projects + classes만 전달받고 서버에서 어노테이션 직접 로드
        if not annotations_data and projects:
            print(f"[DATASET BUILD-FILTERED] compact mode: loading annotations from {len(projects)} projects")
            for project_id, videos in projects.items():
                project_dir = None

                # base_projects_dir/*/<project_id> 구조에서 프로젝트 탐색
                try:
                    user_dirs = sorted([d for d in base_projects_dir.iterdir() if d.is_dir()])
                except Exception:
                    user_dirs = []

                for user_dir in user_dirs:
                    cand = user_dir / project_id
                    if cand.exists() and cand.is_dir():
                        project_dir = cand
                        break

                if not project_dir:
                    continue

                annotations_dir = project_dir / 'annotations'
                if not annotations_dir.exists():
                    continue

                for video_info in videos or []:
                    video_id = video_info.get('video_id') if isinstance(video_info, dict) else None
                    if not video_id:
                        continue

                    video_annotations_dir = annotations_dir / video_id
                    if not video_annotations_dir.exists():
                        continue

                    for json_file in video_annotations_dir.glob('*.json'):
                        if json_file.stem.endswith('.backup') or 'before_fix' in json_file.name or 'discussions' in json_file.name:
                            continue
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                anno_data = json.load(f)
                            annotations_data.append({
                                'project_id': project_id,
                                'video_id': video_id,
                                'annotations': anno_data.get('annotations', {}),
                                'video_name': video_info.get('name', video_id) if isinstance(video_info, dict) else video_id,
                                'project_dir': str(project_dir),
                            })
                        except Exception as e:
                            print(f"[DATASET BUILD-FILTERED] annotation read error {json_file}: {e}")

            print(f"[DATASET BUILD-FILTERED] compact mode loaded annotation files: {len(annotations_data)}")

        if not annotations_data:
            return jsonify({'success': False, 'error': 'No annotations data provided'}), 400

        selected_classes = [c for c in selected_classes if isinstance(c, str) and c.strip()]
        selected_set = set(selected_classes)

        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_dir
        if output_path.exists():
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_path.parent / f"{output_path.name}_{timestamp}"

        for split in ['train', 'val', 'test']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        try:
            tr, vr, ter = map(float, str(split_ratio).split(','))
            total = tr + vr + ter
            if total <= 0:
                raise ValueError('Invalid split ratio')
            train_ratio, val_ratio, test_ratio = tr / total, vr / total, ter / total
        except Exception:
            train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

        korean_to_english = {
            '정상부': 'normal',
            '변형': 'deformation',
            '균열': 'crack',
            '부식': 'corrosion',
            '부식(결절)': 'corrosion_nodule',
            '부식(녹)': 'corrosion_rust',
            '침전물(흙)': 'sediment_soil',
            '침전물(모래)': 'sediment_sand',
            '침전물(부식 생성물)': 'sediment_corrosion',
            '침전물(탈리, 도장재)': 'sediment_coating',
            '침전물(기타)': 'sediment_other',
            '슬라임(물때)': 'slime',
            '논의필요': 'needs_discussion',
            '소실점': 'vanishing_point',
        }

        project_cache = {}

        def resolve_video_path(project_dir: Path, video_id: str):
            pkey = str(project_dir)
            if pkey not in project_cache:
                project_file = project_dir / 'project.json'
                video_map = {}
                class_map = {}
                if project_file.exists():
                    try:
                        with open(project_file, 'r', encoding='utf-8') as f:
                            pj = json.load(f)
                        for idx, cls in enumerate(pj.get('classes', [])):
                            class_map[idx] = cls.get('name', f'class_{idx}')
                        for v in pj.get('videos', []):
                            vid = v.get('video_id')
                            if vid:
                                video_map[vid] = v.get('video_path')
                    except Exception as e:
                        print(f"[DATASET BUILD-FILTERED] project.json parse error: {project_file}: {e}")
                project_cache[pkey] = {'video_map': video_map, 'class_map': class_map}

            meta = project_cache[pkey]
            src_path = meta['video_map'].get(video_id)
            if not src_path:
                return None, meta['class_map']

            src = str(src_path)
            src_path_obj = Path(src_path)
            if 'SAHARA' in src:
                parts = list(src_path_obj.parts)
                i = parts.index('SAHARA')
                rel = Path(*parts[i+1:])
                web_path = Path('/home/intu/nas2_kwater/Videos_web/SAHARA') / rel
                web_path = web_path.with_suffix('.mp4')
            elif '관내시경영상' in src:
                parts = list(src_path_obj.parts)
                i = parts.index('관내시경영상')
                rel = Path(*parts[i+1:])
                web_path = Path('/home/intu/nas2_kwater/Videos_web/관내시경영상') / rel
                web_path = web_path.with_suffix('.mp4')
            else:
                web_path = Path(src.replace('.avi', '.mp4').replace('.AVI', '.mp4'))

            return str(web_path), meta['class_map']

        all_frames = []
        unique_keys = set()
        class_order = list(dict.fromkeys(selected_classes)) if selected_classes else []

        for anno_data in annotations_data:
            project_id = anno_data.get('project_id', '')
            video_id = anno_data.get('video_id', '')
            annotations = anno_data.get('annotations', {}) or {}
            project_dir = Path(anno_data.get('project_dir', ''))

            video_path, class_map = resolve_video_path(project_dir, video_id)
            if not video_path:
                continue
            if not Path(video_path).exists():
                continue

            for frame_num_str, frame_annos in annotations.items():
                if not isinstance(frame_annos, list) or not frame_annos:
                    continue

                try:
                    frame_num = int(frame_num_str)
                except Exception:
                    continue

                filtered_annos = []
                for anno in frame_annos:
                    polygon = anno.get('polygon')
                    if not isinstance(polygon, list) or len(polygon) < 3:
                        continue

                    label = anno.get('label')
                    if not label:
                        class_id = anno.get('class_id')
                        if isinstance(class_id, int):
                            label = class_map.get(class_id)
                        else:
                            try:
                                label = class_map.get(int(class_id))
                            except Exception:
                                label = None

                    if not label:
                        continue
                    if selected_set and label not in selected_set:
                        continue

                    if not selected_set and label not in class_order:
                        class_order.append(label)

                    filtered_annos.append({'label': label, 'polygon': polygon})

                if not filtered_annos:
                    continue

                unique_key = f"{project_id}::{video_id}::{frame_num}"
                if unique_key in unique_keys:
                    continue
                unique_keys.add(unique_key)

                all_frames.append({
                    'project_id': project_id,
                    'video_id': video_id,
                    'video_path': video_path,
                    'frame_num': frame_num,
                    'annotations': filtered_annos,
                })

        if not all_frames:
            return jsonify({'success': False, 'error': 'No frames with selected annotations found'}), 400
        if not class_order:
            return jsonify({'success': False, 'error': 'No classes resolved from annotations'}), 400

        class_to_id = {name: idx for idx, name in enumerate(class_order)}
        yaml_names = [korean_to_english.get(name, name) for name in class_order]

        print(f"[DATASET BUILD-FILTERED] Frames: {len(all_frames)}")
        print(f"[DATASET BUILD-FILTERED] Classes: {class_order}")

        random.shuffle(all_frames)
        train_end = int(len(all_frames) * train_ratio)
        val_end = train_end + int(len(all_frames) * val_ratio)
        train_frames = all_frames[:train_end]
        val_frames = all_frames[train_end:val_end]
        test_frames = all_frames[val_end:]

        def process_frames(frames, split_name):
            saved_count = 0
            frames_by_video = defaultdict(list)
            for frame_data in frames:
                frames_by_video[frame_data['video_path']].append(frame_data)

            def process_video(video_path, video_frames):
                local_count = 0
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"[DATASET BUILD-FILTERED] Cannot open video: {video_path}")
                    return 0
                try:
                    video_frames.sort(key=lambda x: x['frame_num'])
                    for frame_data in video_frames:
                        fnum = frame_data['frame_num']
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            continue

                        image_filename = f"{frame_data['project_id']}_{frame_data['video_id']}_frame{fnum}.jpg"
                        image_path = output_path / split_name / 'images' / image_filename
                        label_path = output_path / split_name / 'labels' / image_filename.replace('.jpg', '.txt')

                        cv2.imwrite(str(image_path), frame)

                        h, w = frame.shape[:2]
                        lines = []
                        for anno in frame_data['annotations']:
                            cid = class_to_id.get(anno['label'])
                            if cid is None:
                                continue
                            coords = []
                            for point in anno['polygon']:
                                try:
                                    x = float(point['x']) / w
                                    y = float(point['y']) / h
                                except Exception:
                                    x = float(point[0]) / w
                                    y = float(point[1]) / h
                                x = max(0.0, min(1.0, x))
                                y = max(0.0, min(1.0, y))
                                coords.append(f"{x:.6f} {y:.6f}")
                            if len(coords) >= 3:
                                lines.append(f"{cid} " + ' '.join(coords))

                        if lines:
                            with open(label_path, 'w', encoding='utf-8') as f:
                                f.write('\\n'.join(lines))
                            local_count += 1
                        else:
                            try:
                                image_path.unlink(missing_ok=True)
                            except Exception:
                                pass
                finally:
                    cap.release()

                return local_count

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_video, vp, vf) for vp, vf in frames_by_video.items()]
                for fut in as_completed(futures):
                    try:
                        saved_count += fut.result()
                    except Exception as e:
                        print(f"[DATASET BUILD-FILTERED] Video worker error: {e}")

            return saved_count

        train_count = process_frames(train_frames, 'train')
        val_count = process_frames(val_frames, 'val')
        test_count = process_frames(test_frames, 'test')

        yaml_content = (
            f"# YOLO Dataset Configuration\\n"
            f"path: {output_path}\\n"
            "train: train/images\\n"
            "val: val/images\\n"
            "test: test/images\\n\\n"
            f"# Number of classes\\n"
            f"nc: {len(yaml_names)}\\n\\n"
            f"# Class names\\n"
            f"names: {yaml_names}\\n"
        )
        with open(output_path / 'data.yaml', 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        info = {
            'created_at': datetime.now().isoformat(),
            'total_frames': len(all_frames),
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'split_ratio': split_ratio,
            'format': 'yolo_segmentation',
            'num_classes': len(class_order),
            'class_names': class_order,
            'class_names_yaml': yaml_names,
            'selected_classes': selected_classes,
            'cache': {
                'cache_dir': str(frame_cache_root),
                'hits': train_hits + val_hits + test_hits,
                'misses': train_misses + val_misses + test_misses,
                'by_split': {
                    'train': {'hit': train_hits, 'miss': train_misses},
                    'val': {'hit': val_hits, 'miss': val_misses},
                    'test': {'hit': test_hits, 'miss': test_misses},
                }
            }
        }
        with open(output_path / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        print(f"[DATASET BUILD-FILTERED] ✅ Complete: {output_path}")

        return jsonify({
            'success': True,
            'output_dir': str(output_path),
            'total_images': train_count + val_count + test_count,
            'train_count': train_count,
            'val_count': val_count,
            'test_count': test_count,
            'classes': class_order,
            'class_names_yaml': yaml_names,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# YOLO Training API
# ============================================================

# 학습 상태 관리
training_state = {
    'is_training': False,
    'job_id': None,
    'progress': {},
    'cancel_requested': False,
    'thread': None
}
training_lock = threading.Lock()


def _run_yolo_training(job_id, config):
    """백그라운드 YOLO 학습 실행"""
    global training_state

    try:
        from ultralytics import YOLO
        import time

        dataset_path = config['dataset_path']
        data_yaml = os.path.join(dataset_path, 'data.yaml')

        if not os.path.exists(data_yaml):
            with training_lock:
                training_state['progress'] = {
                    'status': 'error',
                    'error': f'data.yaml not found: {data_yaml}'
                }
                training_state['is_training'] = False
            return

        # 모델 선택
        model_size = config.get('model_size', 'n')  # n, s, m, l, x
        base_model = config.get('base_model') or f'yolov8{model_size}-seg.pt'

        # 기존 모델에서 이어서 학습 (resume/transfer)
        resume_from = config.get('resume_from')
        if resume_from and os.path.exists(resume_from):
            print(f"[TRAIN] Resuming from: {resume_from}")
            model = YOLO(resume_from)
        else:
            print(f"[TRAIN] Starting from: {base_model}")
            model = YOLO(base_model)

        # 학습 파라미터
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 8)
        img_size = config.get('img_size', 640)
        patience = config.get('patience', 20)
        lr0 = config.get('lr0', 0.01)
        project_name = config.get('project_name', 'pipe_defect')

        # 저장 경로
        script_dir = os.path.dirname(os.path.abspath(__file__))
        runs_dir = os.path.join(script_dir, 'runs')

        with training_lock:
            training_state['progress'] = {
                'status': 'starting',
                'job_id': job_id,
                'config': {
                    'base_model': base_model,
                    'dataset': dataset_path,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'img_size': img_size,
                },
                'epoch': 0,
                'total_epochs': epochs,
                'metrics': {}
            }

        print(f"[TRAIN] Starting training job {job_id}")
        print(f"[TRAIN] Dataset: {dataset_path}")
        print(f"[TRAIN] Model: {base_model}, Epochs: {epochs}, Batch: {batch_size}")

        # 콜백으로 진행률 추적
        def on_train_epoch_end(trainer):
            if training_state['cancel_requested']:
                raise KeyboardInterrupt("Training cancelled by user")

            epoch = trainer.epoch + 1
            metrics = {}
            if hasattr(trainer, 'metrics'):
                for k, v in trainer.metrics.items():
                    try:
                        metrics[k] = float(v)
                    except (TypeError, ValueError):
                        pass

            loss_items = {}
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                loss_names = ['box_loss', 'seg_loss', 'cls_loss', 'dfl_loss']
                for i, name in enumerate(loss_names):
                    if i < len(trainer.loss_items):
                        try:
                            loss_items[name] = float(trainer.loss_items[i])
                        except (TypeError, ValueError):
                            pass

            with training_lock:
                training_state['progress'].update({
                    'status': 'training',
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'metrics': metrics,
                    'loss': loss_items,
                    'percent': round(epoch / epochs * 100, 1)
                })

            print(f"[TRAIN] Epoch {epoch}/{epochs} - Loss: {loss_items} - Metrics: {metrics}")

        def on_train_start(trainer):
            with training_lock:
                training_state['progress']['status'] = 'training'

        # 콜백 등록
        model.add_callback('on_train_epoch_end', on_train_epoch_end)
        model.add_callback('on_train_start', on_train_start)

        # 학습 실행
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            patience=patience,
            lr0=lr0,
            project=runs_dir,
            name=project_name,
            exist_ok=False,
            device=0,
            workers=4,
            verbose=True,
            save=True,
            save_period=10,  # 10 에폭마다 체크포인트
            plots=True,
        )

        # 학습 완료 — best.pt 경로 찾기
        best_model_path = None
        if hasattr(results, 'save_dir'):
            best_path = os.path.join(str(results.save_dir), 'weights', 'best.pt')
            if os.path.exists(best_path):
                best_model_path = best_path

        # 최종 메트릭
        final_metrics = {}
        if results and hasattr(results, 'results_dict'):
            for k, v in results.results_dict.items():
                try:
                    final_metrics[k] = float(v)
                except (TypeError, ValueError):
                    pass

        with training_lock:
            training_state['progress'].update({
                'status': 'completed',
                'epoch': epochs,
                'percent': 100.0,
                'best_model': best_model_path,
                'final_metrics': final_metrics,
                'save_dir': str(results.save_dir) if hasattr(results, 'save_dir') else None
            })
            training_state['is_training'] = False

        print(f"[TRAIN] ✅ Training complete! Best model: {best_model_path}")

    except KeyboardInterrupt:
        with training_lock:
            training_state['progress']['status'] = 'cancelled'
            training_state['is_training'] = False
        print(f"[TRAIN] ⚠️ Training cancelled")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with training_lock:
            training_state['progress'] = {
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            training_state['is_training'] = False
        print(f"[TRAIN] ❌ Training error: {e}")


@app.route('/api/ai/train', methods=['POST'])
def start_training():
    """YOLO 학습 시작"""
    global training_state

    with training_lock:
        if training_state['is_training']:
            return jsonify({
                'success': False,
                'error': 'Training already in progress',
                'job_id': training_state['job_id']
            }), 409

    data = request.json or {}
    dataset_path = data.get('dataset_path')

    if not dataset_path:
        return jsonify({
            'success': False,
            'error': 'dataset_path is required'
        }), 400

    if not os.path.exists(dataset_path):
        return jsonify({
            'success': False,
            'error': f'Dataset not found: {dataset_path}'
        }), 404

    import uuid
    job_id = str(uuid.uuid4())[:8]

    config = {
        'dataset_path': dataset_path,
        'model_size': data.get('model_size', 'n'),
        'base_model': data.get('base_model'),
        'resume_from': data.get('resume_from'),
        'epochs': data.get('epochs', 100),
        'batch_size': data.get('batch_size', 8),
        'img_size': data.get('img_size', 640),
        'patience': data.get('patience', 20),
        'lr0': data.get('lr0', 0.01),
        'project_name': data.get('project_name', 'pipe_defect'),
    }

    with training_lock:
        training_state['is_training'] = True
        training_state['job_id'] = job_id
        training_state['cancel_requested'] = False
        training_state['progress'] = {'status': 'queued', 'job_id': job_id}

    thread = threading.Thread(target=_run_yolo_training, args=(job_id, config), daemon=True)
    thread.start()

    with training_lock:
        training_state['thread'] = thread

    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Training started',
        'config': config
    })


@app.route('/api/ai/train/status', methods=['GET'])
def training_status():
    """학습 진행 상태 조회"""
    with training_lock:
        return jsonify({
            'success': True,
            'is_training': training_state['is_training'],
            'progress': training_state['progress']
        })


@app.route('/api/ai/train/stop', methods=['POST'])
def stop_training():
    """학습 중단"""
    with training_lock:
        if not training_state['is_training']:
            return jsonify({
                'success': False,
                'error': 'No training in progress'
            }), 400

        training_state['cancel_requested'] = True

    return jsonify({
        'success': True,
        'message': 'Cancel requested. Training will stop after current epoch.'
    })


@app.route('/api/ai/models', methods=['GET'])
def list_models():
    """학습된 모델 목록"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(script_dir, 'runs')
    models = []

    # runs/ 하위 디렉토리에서 best.pt 찾기
    if os.path.exists(runs_dir):
        for root, dirs, files in os.walk(runs_dir):
            if 'best.pt' in files:
                best_path = os.path.join(root, 'best.pt')
                # 상위 디렉토리에서 args.yaml 읽기
                train_dir = os.path.dirname(os.path.dirname(best_path))
                args_file = os.path.join(train_dir, 'args.yaml')
                info = {
                    'path': best_path,
                    'name': os.path.basename(train_dir),
                    'size_mb': round(os.path.getsize(best_path) / 1024 / 1024, 1),
                    'created': os.path.getmtime(best_path)
                }
                # args.yaml에서 학습 정보 읽기
                if os.path.exists(args_file):
                    try:
                        import yaml
                        with open(args_file) as f:
                            args = yaml.safe_load(f)
                        info['epochs'] = args.get('epochs')
                        info['imgsz'] = args.get('imgsz')
                        info['model'] = args.get('model')
                        info['data'] = args.get('data')
                    except:
                        pass
                models.append(info)

    # pretrained 모델
    pretrained = os.path.join(script_dir, 'yolov8n-seg.pt')
    if os.path.exists(pretrained):
        models.append({
            'path': pretrained,
            'name': 'yolov8n-seg (pretrained)',
            'size_mb': round(os.path.getsize(pretrained) / 1024 / 1024, 1),
            'created': os.path.getmtime(pretrained),
            'is_pretrained': True
        })

    # 현재 활성 모델
    active_model = None
    if yolo_initialized and yolo_model:
        active_model = str(yolo_model.ckpt_path) if hasattr(yolo_model, 'ckpt_path') else 'unknown'

    return jsonify({
        'success': True,
        'models': sorted(models, key=lambda x: x.get('created', 0), reverse=True),
        'active_model': active_model
    })


@app.route('/api/ai/models/activate', methods=['POST'])
def activate_model():
    """학습된 모델을 추론 모델로 전환"""
    global yolo_model, yolo_initialized

    data = request.json or {}
    model_path = data.get('model_path')

    if not model_path:
        return jsonify({'success': False, 'error': 'model_path required'}), 400

    if not os.path.exists(model_path):
        return jsonify({'success': False, 'error': f'Model not found: {model_path}'}), 404

    try:
        # 기존 모델 해제
        yolo_model = None
        yolo_initialized = False
        torch.cuda.empty_cache()

        success = load_yolo_model(model_path)
        if success:
            return jsonify({
                'success': True,
                'message': f'Model activated: {model_path}'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to load model'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ai/datasets', methods=['GET'])
def list_datasets():
    """빌드된 YOLO 데이터셋 목록"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = []

    for entry in os.scandir(script_dir):
        if entry.is_dir() and entry.name.startswith('pipe_dataset'):
            data_yaml = os.path.join(entry.path, 'data.yaml')
            info_json = os.path.join(entry.path, 'dataset_info.json')

            ds = {
                'name': entry.name,
                'path': entry.path,
            }

            if os.path.exists(info_json):
                try:
                    with open(info_json) as f:
                        info = json.load(f)
                    ds.update({
                        'total_frames': info.get('total_frames', 0),
                        'train_count': info.get('train_count', 0),
                        'val_count': info.get('val_count', 0),
                        'test_count': info.get('test_count', 0),
                        'num_classes': info.get('num_classes', 0),
                        'class_names': info.get('class_names', []),
                        'created_at': info.get('created_at', ''),
                    })
                except:
                    pass
            elif os.path.exists(data_yaml):
                # data.yaml에서 기본 정보 추출
                try:
                    import yaml
                    with open(data_yaml) as f:
                        ydata = yaml.safe_load(f)
                    ds['num_classes'] = ydata.get('nc', 0)
                    ds['class_names'] = ydata.get('names', [])
                except:
                    pass

            # 이미지 수 카운트
            train_imgs = os.path.join(entry.path, 'train', 'images')
            if os.path.exists(train_imgs):
                ds.setdefault('train_count', len(os.listdir(train_imgs)))

            datasets.append(ds)

    return jsonify({
        'success': True,
        'datasets': sorted(datasets, key=lambda x: x['name'], reverse=True)
    })


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
    app.run(host='0.0.0.0', port=int(os.environ.get('GPU_PORT', 5004)), debug=False, threaded=True)
