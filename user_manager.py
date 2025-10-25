#!/usr/bin/env python3
"""
User Manager
사용자 인증 및 세션 관리
"""

import json
import hashlib
import secrets
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict


class UserManager:
    """사용자 관리 클래스"""

    def __init__(self, users_file='users.json'):
        self.users_file = Path(users_file)
        self.users = {}
        self.sessions = {}  # session_id -> {user_id, expires_at}
        self.session_timeout = 3600 * 8  # 8시간

        self.load_users()

    def load_users(self):
        """사용자 데이터 로드"""
        if self.users_file.exists():
            with open(self.users_file, 'r') as f:
                self.users = json.load(f)
        else:
            # 기본 관리자 계정 생성
            self.users = {
                'admin': {
                    'password_hash': self._hash_password('admin123'),
                    'full_name': 'Administrator',
                    'created_at': datetime.now().isoformat(),
                    'role': 'admin'
                }
            }
            self.save_users()

    def save_users(self):
        """사용자 데이터 저장"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)

    def _hash_password(self, password: str) -> str:
        """비밀번호 해싱"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_user(self, user_id: str, password: str, full_name: str = '', role: str = 'user') -> bool:
        """새 사용자 생성"""
        if user_id in self.users:
            return False

        self.users[user_id] = {
            'password_hash': self._hash_password(password),
            'full_name': full_name or user_id,
            'created_at': datetime.now().isoformat(),
            'role': role,
            'projects_dir': f'projects/{user_id}'
        }
        self.save_users()

        # 사용자 프로젝트 폴더 생성
        user_projects_dir = Path(self.users[user_id]['projects_dir'])
        user_projects_dir.mkdir(parents=True, exist_ok=True)

        return True

    def authenticate(self, user_id: str, password: str) -> Optional[str]:
        """사용자 인증 및 세션 생성"""
        if user_id not in self.users:
            return None

        password_hash = self._hash_password(password)
        if self.users[user_id]['password_hash'] != password_hash:
            return None

        # 세션 생성
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            'user_id': user_id,
            'created_at': time.time(),
            'expires_at': time.time() + self.session_timeout,
            'last_activity': time.time()
        }

        return session_id

    def validate_session(self, session_id: str) -> Optional[str]:
        """세션 검증"""
        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # 세션 만료 확인
        if time.time() > session['expires_at']:
            del self.sessions[session_id]
            return None

        # 활동 시간 갱신
        session['last_activity'] = time.time()
        session['expires_at'] = time.time() + self.session_timeout

        return session['user_id']

    def logout(self, session_id: str):
        """로그아웃"""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """사용자 정보 조회"""
        if user_id not in self.users:
            return None

        user = self.users[user_id].copy()
        user.pop('password_hash', None)  # 비밀번호 해시 제거
        user['user_id'] = user_id
        return user

    def list_users(self) -> list:
        """사용자 목록 조회"""
        users = []
        for user_id, user_data in self.users.items():
            info = user_data.copy()
            info.pop('password_hash', None)
            info['user_id'] = user_id
            users.append(info)
        return users

    def update_user(self, user_id: str, new_user_id: str = None, full_name: str = None, role: str = None, password: str = None) -> bool:
        """사용자 정보 업데이트

        Args:
            user_id: 현재 사용자 ID
            new_user_id: 새 사용자 ID (선택사항)
            full_name: 새 이름 (선택사항)
            role: 새 역할 (선택사항)
            password: 새 비밀번호 (선택사항)

        Returns:
            성공 여부
        """
        if user_id not in self.users:
            return False

        # 새 user_id가 제공되고, 기존 ID와 다른 경우
        if new_user_id and new_user_id != user_id:
            # 새 ID가 이미 존재하는지 확인
            if new_user_id in self.users:
                return False

            # 사용자 데이터를 새 ID로 복사
            self.users[new_user_id] = self.users[user_id].copy()

            # 기존 ID 삭제
            del self.users[user_id]

            # 세션의 user_id 업데이트
            for session in self.sessions.values():
                if session['user_id'] == user_id:
                    session['user_id'] = new_user_id

            # 이후 업데이트는 새 ID로 진행
            user_id = new_user_id

        # 업데이트할 필드만 변경
        if full_name is not None:
            self.users[user_id]['full_name'] = full_name

        if role is not None:
            # 역할 검증
            if role not in ['admin', 'user']:
                return False
            self.users[user_id]['role'] = role

        if password is not None:
            self.users[user_id]['password_hash'] = self._hash_password(password)

        self.save_users()
        return True

    def delete_user(self, user_id: str) -> bool:
        """사용자 삭제

        Args:
            user_id: 삭제할 사용자 ID

        Returns:
            성공 여부
        """
        if user_id not in self.users:
            return False

        # 사용자 삭제
        del self.users[user_id]
        self.save_users()

        # 해당 사용자의 활성 세션 모두 제거
        sessions_to_remove = [sid for sid, session in self.sessions.items()
                             if session['user_id'] == user_id]
        for sid in sessions_to_remove:
            del self.sessions[sid]

        return True

    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        current_time = time.time()
        expired = [sid for sid, session in self.sessions.items()
                  if current_time > session['expires_at']]
        for sid in expired:
            del self.sessions[sid]
        return len(expired)
