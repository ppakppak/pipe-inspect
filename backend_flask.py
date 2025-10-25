#!/usr/bin/env python3
"""
Pipe Inspector Backend API (MCP Client Mode)
Flask REST API → MCP Server (GPU 서버)
"""

from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import os
from pathlib import Path
import cv2
from mcp_client import MCPClientSync

app = Flask(__name__)
CORS(app)  # Electron에서 접근 가능하도록

# MCP 서버 경로 설정
MCP_SERVER_SCRIPT = os.getenv(
    'MCP_SERVER_SCRIPT',
    str(Path(__file__).parent / 'mcp-server' / 'server.py')
)

# MCP 클라이언트 초기화 (전역)
mcp_client = None


def get_mcp_client():
    """MCP 클라이언트 가져오기 (싱글톤)"""
    global mcp_client
    if mcp_client is None:
        mcp_client = MCPClientSync(MCP_SERVER_SCRIPT)
        mcp_client.connect()
    return mcp_client


def _load_frame_as_jpeg(video_path: Path, frame_number: int) -> bytes:
    """비디오에서 특정 프레임을 JPEG로 추출"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_number < 0 or frame_number >= total_frames:
        cap.release()
        raise ValueError(f"Frame {frame_number} out of range (total {total_frames})")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to read frame {frame_number} from {video_path}")

    ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")

    return buffer.tobytes()


@app.route('/api/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    try:
        # MCP 서버 연결 확인
        client = get_mcp_client()
        tools = client.list_tools()

        return jsonify({
            'status': 'ok',
            'message': 'Backend is running',
            'mcp_server': 'connected',
            'available_tools': len(tools)
        })
    except Exception as e:
        return jsonify({
            'status': 'degraded',
            'message': 'Backend is running but MCP server unavailable',
            'error': str(e)
        }), 200


@app.route('/api/projects', methods=['GET'])
def list_projects():
    """프로젝트 목록 조회"""
    try:
        client = get_mcp_client()
        result = client.call_tool('list_projects', {})

        if result.get('success'):
            return jsonify({
                'success': True,
                'projects': result.get('projects', [])
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'projects': []
            }), 500

    except Exception as e:
        print(f"Error listing projects: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e),
            'projects': []
        }), 500


@app.route('/api/projects', methods=['POST'])
def create_project():
    """프로젝트 생성"""
    try:
        client = get_mcp_client()
        data = request.json

        # 클래스 문자열을 리스트로 변환
        classes = data['classes']
        if isinstance(classes, str):
            classes = [c.strip() for c in classes.split(',')]

        result = client.call_tool('create_project', {
            'name': data['name'],
            'classes': classes,
            'description': data.get('description', '')
        })

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500

    except Exception as e:
        print(f"Error creating project: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>/frame/<int:frame_number>', methods=['GET'])
def get_video_frame(project_id, video_id, frame_number):
    """비디오 특정 프레임을 JPEG 이미지로 반환"""
    try:
        client = get_mcp_client()
        result = client.call_tool('get_project', {
            'project_id': project_id
        })

        if not result.get('success'):
            status = 404 if 'not found' in result.get('error', '').lower() else 500
            return jsonify(result), status

        project = result.get('project', {})
        project_path = Path(project.get('path', ''))

        video_meta = None
        for video in project.get('videos', []):
            vid = video.get('id') or video.get('video_id')
            if vid == video_id:
                video_meta = video
                break

        if not video_meta:
            return jsonify({'success': False, 'error': 'Video not found'}), 404

        video_path = project_path / video_meta.get('path', '')
        if not video_path.exists():
            return jsonify({
                'success': False,
                'error': f'Video file not found: {video_path}'
            }), 404

        try:
            frame_bytes = _load_frame_as_jpeg(video_path, frame_number)
        except ValueError as ve:
            return jsonify({'success': False, 'error': str(ve)}), 400

        return Response(frame_bytes, mimetype='image/jpeg')

    except Exception as e:
        print(f"Error getting video frame: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """프로젝트 상세 정보 조회"""
    try:
        client = get_mcp_client()
        result = client.call_tool('get_project', {
            'project_id': project_id
        })

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 404 if 'not found' in result.get('error', '').lower() else 500

    except Exception as e:
        print(f"Error getting project: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos', methods=['POST'])
def add_video(project_id):
    """프로젝트에 비디오 추가"""
    try:
        client = get_mcp_client()
        data = request.json

        result = client.call_tool('add_video', {
            'project_id': project_id,
            'video_path': data['video_path']
        })

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 404 if 'not found' in result.get('error', '').lower() else 500

    except Exception as e:
        print(f"Error adding video: {e}")
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
        client = get_mcp_client()
        result = client.call_tool('get_project', {
            'project_id': project_id
        })

        if not result.get('success'):
            status = 404 if 'not found' in result.get('error', '').lower() else 500
            return jsonify(result), status

        project = result.get('project', {})
        for video in project.get('videos', []):
            vid = video.get('id') or video.get('video_id')
            if vid == video_id:
                video_info = {
                    'id': vid,
                    'filename': video.get('filename', ''),
                    'path': video.get('path', ''),
                    'total_frames': video.get('total_frames', video.get('frame_count', 0)),
                    'frame_count': video.get('frame_count', video.get('total_frames', 0)),
                    'status': video.get('status', ''),
                    'annotations': video.get('annotations', 0)
                }
                return jsonify({'success': True, 'video': video_info})

        return jsonify({
            'success': False,
            'error': 'Video not found'
        }), 404

    except Exception as e:
        print(f"Error getting video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/projects/<project_id>/videos/<video_id>', methods=['DELETE'])
def remove_video(project_id, video_id):
    """프로젝트에서 비디오 제거"""
    try:
        client = get_mcp_client()

        result = client.call_tool('remove_video', {
            'project_id': project_id,
            'video_id': video_id
        })

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 404 if 'not found' in result.get('error', '').lower() else 500

    except Exception as e:
        print(f"Error removing video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/ai/initialize', methods=['POST'])
def initialize_ai():
    """AI 모델 초기화 (MCP 모드에서는 미지원)"""
    return jsonify({
        'success': False,
        'error': 'AI inference is not available in MCP client mode.',
        'details': '원격 GPU 서버 또는 프록시 백엔드를 사용하세요.'
    }), 501


@app.route('/api/ai/inference', methods=['POST'])
def run_inference():
    """AI 추론 실행 (MCP 모드에서는 미지원)"""
    return jsonify({
        'success': False,
        'error': 'AI inference is not available in MCP client mode.'
    }), 501


@app.route('/api/projects/<project_id>/statistics', methods=['GET'])
def get_statistics(project_id):
    """프로젝트 통계 정보 조회"""
    try:
        client = get_mcp_client()

        result = client.call_tool('get_project_statistics', {
            'project_id': project_id
        })

        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 404 if 'not found' in result.get('error', '').lower() else 500

    except Exception as e:
        print(f"Error getting statistics: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/test', methods=['GET'])
def test():
    """테스트 엔드포인트"""
    import sys
    return jsonify({
        'message': 'Hello from Python Backend (MCP Client Mode)!',
        'python_version': sys.version,
        'mcp_server_script': MCP_SERVER_SCRIPT
    })


@app.route('/api/mcp/tools', methods=['GET'])
def list_mcp_tools():
    """MCP 서버의 사용 가능한 도구 목록"""
    try:
        client = get_mcp_client()
        tools = client.list_tools()

        tools_info = []
        for tool in tools:
            tools_info.append({
                'name': tool.name,
                'description': tool.description,
            })

        return jsonify({
            'success': True,
            'tools': tools_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting Pipe Inspector Backend (MCP Client Mode)...")
    print(f"📡 API Server: http://localhost:5003")
    print(f"🔌 MCP Server Script: {MCP_SERVER_SCRIPT}")

    try:
        app.run(host='0.0.0.0', port=5003, debug=True)
    finally:
        # 종료 시 MCP 클라이언트 정리
        if mcp_client:
            mcp_client.disconnect()
