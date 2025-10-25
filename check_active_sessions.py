#!/usr/bin/env python3
"""
활성 세션 확인 스크립트
"""
import json
import time
from datetime import datetime
from user_manager import UserManager

def main():
    user_manager = UserManager()

    print("=" * 60)
    print("현재 활성 세션 목록")
    print("=" * 60)

    active_sessions = []
    current_time = time.time()

    for session_id, session_data in user_manager.sessions.items():
        user_id = session_data['user_id']
        created_at = datetime.fromtimestamp(session_data['created_at'])
        expires_at = datetime.fromtimestamp(session_data['expires_at'])
        last_activity = datetime.fromtimestamp(session_data['last_activity'])

        # 만료되지 않은 세션만
        if current_time < session_data['expires_at']:
            user_info = user_manager.get_user_info(user_id)
            active_sessions.append({
                'session_id': session_id[:12] + '...',
                'user_id': user_id,
                'role': user_info.get('role', 'unknown'),
                'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'last_activity': last_activity.strftime('%Y-%m-%d %H:%M:%S'),
                'expires_at': expires_at.strftime('%Y-%m-%d %H:%M:%S')
            })

    if not active_sessions:
        print("\n현재 활성 세션이 없습니다.\n")
    else:
        print(f"\n총 {len(active_sessions)}개의 활성 세션\n")
        for i, session in enumerate(active_sessions, 1):
            print(f"{i}. 사용자: {session['user_id']} ({session['role']})")
            print(f"   세션 ID: {session['session_id']}")
            print(f"   생성 시간: {session['created_at']}")
            print(f"   마지막 활동: {session['last_activity']}")
            print(f"   만료 시간: {session['expires_at']}")
            print()

    # 만료된 세션 정리
    expired_count = user_manager.cleanup_expired_sessions()
    if expired_count > 0:
        print(f"만료된 세션 {expired_count}개 정리됨")

    print("=" * 60)

if __name__ == '__main__':
    main()
