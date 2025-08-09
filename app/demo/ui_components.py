"""
맛집 검색 LLM 에이전트 UI 컴포넌트 모듈
"""

import gradio as gr
from typing import Any, Callable
from .config import config, ui_messages
from .session import SessionManager
from .utils import create_session_id, format_timestamp


def create_chat_interface(chat_fn: Callable) -> tuple[Any, Any, Any]:
    """
    채팅 인터페이스 생성
    
    Args:
        chat_fn: 채팅 처리 함수
        
    Returns:
        tuple: (chat_interface, session_info, search_history_info)
    """
    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        examples=ui_messages.chat_examples,
        cache_examples=False,
        chatbot=gr.Chatbot(height=config.chatbot_height)
    )
    
    # 세션 상태 표시
    with gr.Row():
        with gr.Column(scale=2):
            session_info = gr.Markdown("### 세션 정보\n활성 세션 수: 0")
        with gr.Column(scale=1):
            refresh_btn = gr.Button("세션 정보 새로고침", size="sm")

    with gr.Row():
        search_history_info = gr.Textbox(
            label="현재 세션의 검색 히스토리",
            lines=config.search_history_lines,
            max_lines=config.search_history_max_lines,
            interactive=False,
            placeholder=ui_messages.no_search_history
        )

    def update_session_info(request: gr.Request = None):
        from .demo import session_manager  # 순환 import 방지
        
        active_sessions = session_manager.get_active_sessions_count()
        session_info_text = f"### 세션 정보\n활성 세션 수: {active_sessions}"
        
        # 현재 사용자 세션 정보
        session_id = None
        if request:
            session_id = create_session_id(request.client.host)
        
        search_history_text = ui_messages.no_search_history
        if session_id:
            session = session_manager.get_session(session_id)
            if session and session.search_history:
                search_history_text = ""
                for i, search in enumerate(session.search_history, 1):
                    timestamp = format_timestamp(search['timestamp'])
                    search_history_text += f"{i}. [{timestamp}] \"{search['query']}\"\n"
                    search_history_text += f"   컨텍스트 길이: {search['context_length']} 문자\n\n"
        
        return session_info_text, search_history_text

    refresh_btn.click(
        fn=update_session_info,
        outputs=[session_info, search_history_info]
    )
    
    return chat_interface, session_info, search_history_info


def create_structure_query_tab(test_fn: Callable) -> None:
    """쿼리 구조화 테스트 탭 생성"""
    gr.Markdown("### 자연어 쿼리를 구조화된 JSON으로 변환하는 모듈을 테스트합니다.")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="테스트 쿼리",
                placeholder=f"예: {ui_messages.test_structure_query}",
                value=ui_messages.test_structure_query
            )
            structure_btn = gr.Button("쿼리 구조화 테스트", variant="primary")
        
        with gr.Column():
            structure_output = gr.Code(
                label="구조화된 쿼리 결과",
                language="json",
                lines=15
            )
    
    structure_btn.click(
        fn=test_fn,
        inputs=[query_input],
        outputs=[structure_output]
    )


def create_search_test_tab(test_fn: Callable) -> None:
    """검색 모듈 테스트 탭 생성"""
    gr.Markdown("### 쿼리 재작성 → 임베딩 → Elasticsearch 검색 파이프라인을 테스트합니다.")
    
    with gr.Row():
        with gr.Column():
            search_query_input = gr.Textbox(
                label="테스트 쿼리",
                placeholder=f"예: {ui_messages.test_search_query}",
                value=ui_messages.test_search_query
            )
            search_btn = gr.Button("검색 모듈 테스트", variant="primary")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            structured_query_output = gr.Code(
                label="1단계: 구조화된 쿼리",
                language="json",
            )
        with gr.Column(scale=3):
            es_query_output = gr.Code(
                label="2단계: Elasticsearch 쿼리",
                language="json",
            )
        with gr.Column(scale=5):
            search_results_output = gr.Code(
                label="3단계: 검색 결과",
            )
    
    search_btn.click(
        fn=test_fn,
        inputs=[search_query_input],
        outputs=[structured_query_output, es_query_output, search_results_output]
    )


def create_relevance_test_tab(
    relevance_fn: Callable, 
    search_results_fn: Callable
) -> None:
    """연관성 평가 테스트 탭 생성"""
    gr.Markdown("### 실제 Elasticsearch 검색 결과에 대한 연관성 평가와 요약 정보를 표시합니다.")
    
    with gr.Row():
        with gr.Column():
            relevance_query_input = gr.Textbox(
                label="테스트 쿼리",
                placeholder=f"예: {ui_messages.test_relevance_query}",
                value=ui_messages.test_relevance_query
            )
            relevance_btn = gr.Button("연관성 평가 실행", variant="primary")
            
            # 연관성 평가 결과 (왼쪽 아래)
            relevance_output = gr.Textbox(
                label="연관성 평가 결과",
                lines=30,
                max_lines=35,
            )
        
        with gr.Column():
            # 검색 결과 요약 (오른쪽)
            search_results_output = gr.Textbox(
                label="검색 결과 요약",
                lines=50,
                max_lines=50,
            )
    
    relevance_btn.click(
        fn=relevance_fn,
        inputs=[relevance_query_input],
        outputs=[relevance_output]
    )
    
    relevance_btn.click(
        fn=search_results_fn,
        inputs=[relevance_query_input],
        outputs=[search_results_output]
    )


def create_admin_dashboard(
    test_structure_query_fn: Callable,
    test_search_module_fn: Callable,
    get_relevance_evaluation_fn: Callable,
    get_search_results_summary_fn: Callable
) -> None:
    """어드민 대시보드 생성"""
    gr.Markdown("## 각 모듈의 결과를 확인하고 디버깅할 수 있는 인터페이스입니다.")
    
    with gr.Tabs():
        # 쿼리 재작성 모듈 테스트
        with gr.Tab("🔄 쿼리 재작성"):
            create_structure_query_tab(test_structure_query_fn)
        
        # 검색 모듈 테스트
        with gr.Tab("🔍 검색"):
            create_search_test_tab(test_search_module_fn)
        
        # 연관성 평가 모듈 테스트
        with gr.Tab("⚖️ 연관성 평가"):
            create_relevance_test_tab(
                get_relevance_evaluation_fn,
                get_search_results_summary_fn
            )
    
    gr.Markdown("---")
    gr.Markdown("💡 **사용법**: 각 탭에서 다른 쿼리를 입력해서 각 모듈의 동작을 확인할 수 있습니다.")