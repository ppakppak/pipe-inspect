#!/usr/bin/env python3
"""
Pipe Inspector Backend API
Flask REST API for Electron frontend
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import sys
import os

# 기존 프로젝트 경로 추가
sys.path.insert(0, '/home/ppak/SynologyDrive/ykpark/linux_devel/ground_sam/Grounded-Segment-Anything')

app = Flask(__name__)
CORS(app)  # Electron에서 접근 가능하도록

@app.route('/api/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        'status': 'ok',
        'message': 'Backend is running'
    })

@app.route('/api/projects', methods=['GET'])
def list_projects():
    """프로젝트 목록 조회"""
    try:
        from project_manager import ProjectManager
        pm = ProjectManager()
        projects_list = pm.list_projects()

        # 프로젝트 정보를 dict 형태로 변환
        projects_data = []
        for p in projects_list:
            # Project 객체인 경우
            if hasattr(p, 'id'):
                projects_data.append({
                    'id': p.id,
                    'name': p.name,
                    'path': str(p.project_dir)
                })
            # dict인 경우
            elif isinstance(p, dict):
                projects_data.append({
                    'id': p.get('id', 'unknown'),
                    'name': p.get('name', 'Unknown'),
                    'path': p.get('path', 'Unknown')
                })

        return jsonify({
            'success': True,
            'projects': projects_data
        })
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
        from project_manager import ProjectManager
        data = request.json
        pm = ProjectManager()
        project = pm.create_project(
            name=data['name'],
            classes=data['classes'].split(',')
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
    """프로젝트 상세 정보 조회"""
    try:
        from project_manager import ProjectManager
        pm = ProjectManager()

        # 프로젝트 로드
        projects = pm.list_projects()
        project = None
        for p in projects:
            if hasattr(p, 'id') and p.id == project_id:
                project = p
                break

        if not project:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 통계 정보 가져오기
        stats = project.get_statistics()

        # 비디오 목록 가져오기
        videos = []
        try:
            # videos가 dict인 경우
            if hasattr(project, 'videos') and isinstance(project.videos, dict):
                for video_id, video_info in project.videos.items():
                    videos.append({
                        'id': video_id,
                        'filename': video_info.get('filename', 'Unknown'),
                        'path': str(video_info.get('path', '')),
                        'frame_count': video_info.get('frame_count', 0),
                        'annotations': video_info.get('annotations', 0)
                    })
            # videos가 list인 경우
            elif hasattr(project, 'videos') and isinstance(project.videos, list):
                for video_info in project.videos:
                    videos.append({
                        'id': video_info.get('id', ''),
                        'filename': video_info.get('filename', 'Unknown'),
                        'path': str(video_info.get('path', '')),
                        'frame_count': video_info.get('frame_count', 0),
                        'annotations': video_info.get('annotations', 0)
                    })
        except Exception as e:
            print(f"Error loading videos: {e}")
            import traceback
            traceback.print_exc()

        return jsonify({
            'success': True,
            'project': {
                'id': project.id,
                'name': project.name,
                'path': str(project.project_dir),
                'classes': project.classes,
                'description': project.description if hasattr(project, 'description') else '',
                'stats': stats,
                'videos': videos
            }
        })
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
        from project_manager import ProjectManager
        pm = ProjectManager()
        data = request.json

        # 프로젝트 로드
        projects = pm.list_projects()
        project = None
        for p in projects:
            if hasattr(p, 'id') and p.id == project_id:
                project = p
                break

        if not project:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 비디오 추가
        video_id = project.add_video(data['video_path'])

        return jsonify({
            'success': True,
            'video_id': video_id
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/projects/<project_id>/videos/<video_id>', methods=['DELETE'])
def remove_video(project_id, video_id):
    """프로젝트에서 비디오 제거"""
    try:
        from project_manager import ProjectManager
        pm = ProjectManager()

        # 프로젝트 로드
        projects = pm.list_projects()
        project = None
        for p in projects:
            if hasattr(p, 'id') and p.id == project_id:
                project = p
                break

        if not project:
            return jsonify({
                'success': False,
                'error': 'Project not found'
            }), 404

        # 비디오 제거
        project.remove_video(video_id)

        return jsonify({
            'success': True
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test', methods=['GET'])
def test():
    """테스트 엔드포인트"""
    return jsonify({
        'message': 'Hello from Python Backend!',
        'python_version': sys.version
    })

if __name__ == '__main__':
    print("🚀 Starting Pipe Inspector Backend...")
    print("📡 API Server: http://localhost:5003")
    app.run(host='0.0.0.0', port=5003, debug=True)
