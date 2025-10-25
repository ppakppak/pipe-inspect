#!/usr/bin/env python3
"""
Project Manager
GPU 서버에서 프로젝트 메타데이터를 관리하기 위한 경량 구현.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _default_projects_base_dir() -> Path:
    """환경 변수 또는 기본 경로에서 프로젝트 루트를 결정."""
    base = os.getenv('PROJECTS_BASE_DIR', '/home/intu/nas2/k_water/pipe_inspector_data')
    base_path = Path(base).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


PROJECTS_BASE_DIR = _default_projects_base_dir()


@dataclass
class Project:
    """프로젝트 메타데이터 컨테이너"""

    id: str
    name: str
    project_dir: Path
    classes: List[str] = field(default_factory=list)
    description: str = ""
    user_id: Optional[str] = None
    videos: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def get_statistics(self) -> Dict[str, int]:
        """간단한 통계 정보 계산"""
        videos = self.videos or []
        total_annotations = 0
        annotated_frames = 0
        annotated_videos = 0

        for video in videos:
            annotations = video.get('annotations', {})
            if isinstance(annotations, dict):
                frame_counts = []
                for frame_data in annotations.values():
                    if isinstance(frame_data, dict):
                        frame_counts.append(len(frame_data.get('regions', [])))
                    elif isinstance(frame_data, list):
                        frame_counts.append(len(frame_data))
                    else:
                        frame_counts.append(0)
                annotated_frames += sum(1 for count in frame_counts if count > 0)
                frame_total = sum(frame_counts)
                if frame_total > 0:
                    annotated_videos += 1
                    total_annotations += frame_total
            elif isinstance(annotations, list):
                # 프레임 단위가 아닌 리스트 구조
                count = len(annotations)
                if count > 0:
                    annotated_videos += 1
                    annotated_frames += 1
                    total_annotations += count

        datasets = self.metadata.get('datasets', [])
        datasets_count = len(datasets) if isinstance(datasets, list) else 0

        return {
            'total_videos': len(videos),
            'annotated_videos': annotated_videos,
            'total_annotations': total_annotations,
            'annotated_frames': annotated_frames,
            'datasets': datasets_count
        }


class ProjectManager:
    """프로젝트 디렉터리를 스캔하고 조작하는 매니저"""

    def __init__(self, base_dir: Optional[Path | str] = None):
        self.base_dir = Path(base_dir).expanduser().resolve() if base_dir else PROJECTS_BASE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _load_project(self, project_json: Path) -> Optional[Project]:
        try:
            with open(project_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[ProjectManager] Failed to load project: {project_json} ({exc})")
            return None

        project_dir = project_json.parent
        project_id = data.get('id') or project_dir.name
        name = data.get('name') or project_id
        project = Project(
            id=project_id,
            name=name,
            project_dir=project_dir,
            classes=data.get('classes', []),
            description=data.get('description', ''),
            user_id=data.get('user_id'),
            videos=data.get('videos', []),
            metadata=data
        )
        return project

    def list_projects(self) -> List[Project]:
        """project.json 파일을 재귀적으로 찾아 프로젝트 목록 반환"""
        projects: List[Project] = []
        for project_json in self.base_dir.rglob('project.json'):
            project = self._load_project(project_json)
            if project:
                projects.append(project)

        projects.sort(key=lambda p: p.metadata.get('created_at', ''), reverse=True)
        return projects

    def _generate_project_id(self, name: str) -> str:
        """프로젝트 이름 기반의 고유 ID 생성"""
        safe_name = re.sub(r'[^0-9a-zA-Z가-힣_-]+', '_', name).strip('_')
        if not safe_name:
            safe_name = 'project'
        timestamp = int(time.time())
        return f"{safe_name}_{timestamp}"

    def create_project(self, name: str, classes: List[str], description: str = "", user_id: Optional[str] = None) -> Project:
        """프로젝트 생성 후 Project 객체 반환"""
        project_id = self._generate_project_id(name)
        project_dir = self.base_dir / (user_id or '') / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / 'videos').mkdir(exist_ok=True)

        project_data = {
            'id': project_id,
            'name': name,
            'classes': classes,
            'description': description,
            'user_id': user_id,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'videos': []
        }

        project_json = project_dir / 'project.json'
        with open(project_json, 'w', encoding='utf-8') as f:
            json.dump(project_data, f, indent=2, ensure_ascii=False)

        return self._load_project(project_json)
