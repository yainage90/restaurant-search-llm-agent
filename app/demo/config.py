"""
맛집 검색 LLM 에이전트 설정 관리 모듈
"""

from dataclasses import dataclass
import os


@dataclass
class AppConfig:
    """애플리케이션 설정"""
    
    # 서버 설정
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    
    # Elasticsearch 설정
    elasticsearch_index: str = "restaurants"
    
    # 세션 설정
    session_max_age_hours: int = 24
    session_cleanup_interval: int = 10  # 분
    
    # 검색 설정
    similarity_threshold: float = 0.7
    max_search_history: int = 3
    max_chat_history: int = 5
    
    # UI 설정
    chatbot_height: str = "80vh"
    search_history_lines: int = 5
    search_history_max_lines: int = 10
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """환경 변수에서 설정을 로드"""
        return cls(
            server_name=os.getenv("GRADIO_SERVER_NAME", cls.server_name),
            server_port=int(os.getenv("GRADIO_SERVER_PORT", cls.server_port)),
            share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
            elasticsearch_index=os.getenv("ELASTICSEARCH_INDEX", cls.elasticsearch_index),
            session_max_age_hours=int(os.getenv("SESSION_MAX_AGE_HOURS", cls.session_max_age_hours)),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", cls.similarity_threshold)),
        )


@dataclass
class UIMessages:
    """UI에 표시되는 메시지들"""
    
    # 채팅 예제
    chat_examples = [
        "강남역 근처 맛있는 한식당 추천해줘",
        "데이트하기 좋은 이태리 레스토랑이 ",
        "맥도날드 vs 버거킹 뭐먹지?",
        "아이데리고 가기좋은 강남 한식당",
        "강남 보승회관 주차돼?",
    ]
    
    # 테스트 쿼리들
    test_nlu_query = "강남역 주차되는 일식집"
    test_search_query = "마포 진대감 주차되나요?"
    test_relevance_query = "강남역 주차되는 일식집"
    
    # 플레이스홀더 메시지들
    no_search_history = "검색 기록이 없습니다."
    no_search_results = "검색 결과가 없습니다."
    
    # 에러 메시지들
    error_prefix = "오류 발생: "
    parsing_error = "LLM 응답 파싱 실패"
    search_decision_error = "검색 필요성 판단 중 오류"


# 전역 설정 인스턴스
config = AppConfig.from_env()
ui_messages = UIMessages()