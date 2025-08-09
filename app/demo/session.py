"""
맛집 검색 LLM 에이전트 세션 관리 모듈
"""

from datetime import datetime
from typing import Optional
from .config import config


class ConversationSession:
    """개별 사용자 세션의 대화 상태를 관리하는 클래스"""
    
    def __init__(self, session_id: str = None):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.context: Optional[str] = None  # 검색된 레스토랑 정보
        self.search_history: list[dict] = []  # 검색 기록
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
    
    def update_context(self, query: str, restaurant_context: str) -> None:
        """새로운 검색 결과로 컨텍스트 업데이트"""
        self.context = restaurant_context
        self.search_history.append({
            "query": query,
            "timestamp": datetime.now(),
            "context_length": len(restaurant_context) if restaurant_context else 0
        })
        self.last_activity = datetime.now()
    
    def get_context(self) -> str:
        """현재 컨텍스트 반환"""
        return self.context or ""
    
    def has_context(self) -> bool:
        """컨텍스트가 존재하는지 확인"""
        return self.context is not None
    
    def get_search_count(self) -> int:
        """총 검색 횟수 반환"""
        return len(self.search_history)
    
    def get_recent_searches(self, limit: int = None) -> list[dict]:
        """최근 검색 기록 반환"""
        if limit is None:
            limit = config.max_search_history
        return self.search_history[-limit:]
    
    def is_expired(self, max_age_hours: int = None) -> bool:
        """세션이 만료되었는지 확인"""
        if max_age_hours is None:
            max_age_hours = config.session_max_age_hours
        
        now = datetime.now()
        age_hours = (now - self.last_activity).total_seconds() / 3600
        return age_hours > max_age_hours


class SessionManager:
    """세션들을 관리하는 매니저 클래스"""
    
    def __init__(self):
        self.sessions: dict[str, ConversationSession] = {}
        self._last_cleanup = datetime.now()
    
    def get_or_create_session(self, session_id: str = None) -> ConversationSession:
        """세션을 가져오거나 새로 생성"""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_activity = datetime.now()
            return session
        
        session = ConversationSession(session_id)
        self.sessions[session.session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """특정 세션 가져오기"""
        return self.sessions.get(session_id)
    
    def cleanup_old_sessions(self, max_age_hours: int = None) -> int:
        """오래된 세션들을 정리"""
        if max_age_hours is None:
            max_age_hours = config.session_max_age_hours
        
        # 정리 간격 확인 (너무 자주 실행되지 않도록)
        now = datetime.now()
        minutes_since_cleanup = (now - self._last_cleanup).total_seconds() / 60
        if minutes_since_cleanup < config.session_cleanup_interval:
            return 0
        
        to_remove = []
        for session_id, session in self.sessions.items():
            if session.is_expired(max_age_hours):
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.sessions[session_id]
        
        self._last_cleanup = now
        return len(to_remove)
    
    def get_active_sessions_count(self) -> int:
        """활성 세션 수 반환"""
        return len(self.sessions)
    
    def get_session_stats(self) -> dict:
        """세션 통계 정보 반환"""
        if not self.sessions:
            return {
                "total_sessions": 0,
                "total_searches": 0,
                "avg_searches_per_session": 0.0
            }
        
        total_searches = sum(session.get_search_count() for session in self.sessions.values())
        
        return {
            "total_sessions": len(self.sessions),
            "total_searches": total_searches,
            "avg_searches_per_session": total_searches / len(self.sessions)
        }