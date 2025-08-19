"""
맛집 검색 LLM 에이전트의 통합 인터페이스
사용자용 채팅 인터페이스와 어드민 대시보드를 하나로 통합한 Gradio 앱
"""

import gradio as gr
from dotenv import load_dotenv
from typing import Optional
from app.retrieve.search import search, search_restaurants
from app.generation.generation import generate
from app.retrieve.relevance import grade_relevance
from app.demo.config import config, ui_messages
from app.demo.session import SessionManager
from app.demo.utils import (
    handle_exceptions, 
    parse_json_response, 
    safe_format_json, 
    calculate_keyword_similarity,
    create_session_id,
    generate_decision
)
from app.demo.ui_components import (
    create_chat_interface,
    create_admin_dashboard
)

load_dotenv()


# 전역 세션 매니저
session_manager = SessionManager()


@handle_exceptions(
    default_return={
        "need_search": False, 
        "reason": ui_messages.search_decision_error
    }
)
def should_perform_new_search(
    current_message: str, 
    search_history: list[dict], 
    chat_history: list[list[str]]
) -> dict[str, str | bool]:
    """
    새로운 검색이 필요한지 LLM으로 판단하는 함수
    
    Args:
        current_message: 현재 사용자 메시지
        search_history: 이전 검색 기록
        chat_history: 대화 기록
        
    Returns:
        dict: 검색 필요성 판단 결과
    """
    # 검색 기록을 텍스트로 변환
    search_history_text = ""
    if search_history:
        search_history_text = "이전 검색 기록:\n"
        recent_searches = search_history[-config.max_search_history:]
        for i, search in enumerate(recent_searches, 1):
            timestamp = search['timestamp'].strftime('%H:%M')
            search_history_text += f"{i}. \"{search['query']}\" (검색시간: {timestamp})\n"
    
    # 대화 기록을 텍스트로 변환
    chat_history_text = ""
    if chat_history:
        chat_history_text = "최근 대화:\n"
        recent_chat = chat_history[-config.max_chat_history:]
        for user_msg, bot_msg in recent_chat:
            chat_history_text += f"사용자: {user_msg}\n"
            chat_history_text += f"어시스턴트: {bot_msg}\n"
    
    prompt = f"""사용자의 현재 메시지를 분석하여 새로운 맛집 검색이 필요한지 판단해주세요.

현재 사용자 메시지: "{current_message}"

{search_history_text}

{chat_history_text}

다음과 같은 경우 새로운 검색이 필요합니다:
1. 완전히 다른 지역이나 음식 종류를 요청하는 경우
2. 기존 검색 결과에 만족하지 못하고 다른 옵션을 원하는 경우
3. 이전 검색과 전혀 관련 없는 새로운 맛집 정보를 요청하는 경우
4. 검색 기록이 없는 상태에서 특정 식당명이나 구체적인 맛집 정보를 요청하는 경우

다음과 같은 경우는 기존 검색 결과로 답변 가능합니다:
1. 기존 검색 기록이 있고 그 검색 결과에 포함된 특정 식당에 대한 추가 질문 (메뉴, 영업시간, 가격, 위치, 예약 방법 등)
2. 기존 검색 기록이 있고 그 추천 맛집 중 선택이나 비교를 요청하는 경우  
3. 일반적인 대화나 감사 인사
4. 기존 검색 기록이 있고 그 검색 결과 내에서 답변할 수 있는 질문


응답 형식:
{{
    "need_search": true/false,
    "reason": "판단 근거를 한 문장으로"
}}"""

    response = generate_decision(prompt)
    parsed_result = parse_json_response(response)
    
    if "error" in parsed_result:
        return {
            "need_search": False,
            "reason": parsed_result["error"]
        }
    
    return {
        "need_search": parsed_result.get("need_search", False),
        "reason": parsed_result.get("reason", "판단 불가")
    }


def is_similar_query(
    new_query: str, 
    search_history: list[dict], 
    similarity_threshold: Optional[float] = None
) -> bool:
    """
    새 쿼리가 기존 검색과 유사한지 확인
    
    Args:
        new_query: 새로운 검색 쿼리
        search_history: 검색 기록
        similarity_threshold: 유사도 임계값
        
    Returns:
        bool: 유사한 검색이 있으면 True
    """
    if not search_history:
        return False
    
    if similarity_threshold is None:
        similarity_threshold = config.similarity_threshold
    
    recent_searches = search_history[-config.max_search_history:]
    
    for search in recent_searches:
        similarity = calculate_keyword_similarity(new_query, search['query'])
        if similarity >= similarity_threshold:
            return True
    
    return False

def chat_fn(
    message: str, 
    history: list[list[str]], 
    request: Optional[gr.Request] = None
) -> str:
    """
    채팅 처리 함수 - 스마트 검색 로직을 포함한 세션 기반 상태 관리
    
    Args:
        message: 사용자의 현재 메시지
        history: 이전 대화 기록
        request: Gradio 요청 객체 (세션 ID 추출용)
        
    Returns:
        str: 응답 메시지
    """
    # 세션 ID 추출 및 세션 관리
    session_id = None
    if request:
        session_id = create_session_id(request.client.host)
    
    session = session_manager.get_or_create_session(session_id)
    session_manager.cleanup_old_sessions()

    if not history or not session.has_context():
        # 첫 대화: 검색 필요성 판단 후 처리
        return _handle_first_chat(message, session)
    else:
        # 후속 대화: 스마트 검색 로직 적용
        return _handle_follow_up_chat(message, history, session)


def _handle_first_chat(message: str, session) -> str:
    """첫 번째 대화 처리 - 검색 필요성 판단"""
    # 빈 기록으로 검색 필요성 판단
    search_decision = should_perform_new_search(message, [], [])
    print(f'[Session {session.session_id}] 첫 대화 검색 필요성 판단: {search_decision}')
    
    if search_decision["need_search"]:
        # 검색이 필요한 경우
        return _perform_new_search(message, session)
    else:
        # 검색이 불필요한 경우 - 일반적인 응답 생성
        try:
            bot_response = generate(query="", context=f"사용자 메시지: {message}")
            print(f'[Session {session.session_id}] 첫 대화 - 검색 없이 응답')
            return bot_response
        except Exception as e:
            print(f'[Session {session.session_id}] 응답 생성 오류: {e}')
            return f"{ui_messages.error_prefix}{str(e)}"


def _handle_follow_up_chat(message: str, history: list[list[str]], session) -> str:
    """후속 대화 처리"""
    # 검색 필요성 판단
    search_decision = should_perform_new_search(message, session.search_history, history)
    print(f'[Session {session.session_id}] 검색 필요성 판단: {search_decision}')
    
    if search_decision["need_search"]:
        # 중복 검색 방지
        if is_similar_query(message, session.search_history):
            print(f'[Session {session.session_id}] 유사한 검색으로 판단, 기존 컨텍스트 사용')
        else:
            # 새로운 검색 수행
            return _perform_new_search(message, session)
    
    # 기존 컨텍스트로 응답 생성
    return _generate_context_response(message, history, session)


def _perform_new_search(query: str, session) -> str:
    """새로운 검색 수행"""
    try:
        print(f'[Session {session.session_id}] 새로운 검색 수행 - Query: {query}')
        restaurant_context = search(query)
        session.update_context(query, restaurant_context)
        
        bot_response = generate(query, restaurant_context)
        return bot_response
    except Exception as e:
        print(f'[Session {session.session_id}] 새 검색 오류: {e}')
        return f"{ui_messages.error_prefix}{str(e)}"


def _generate_context_response(message: str, history: list[list[str]], session) -> str:
    """기존 컨텍스트를 활용한 응답 생성"""
    try:
        chat_history_prompt = ""
        for user_msg, bot_msg in history:
            chat_history_prompt += f"사용자: {user_msg}\n"
            chat_history_prompt += f"어시스턴트: {bot_msg}\n"
        
        chat_history_prompt += f"사용자: {message}\n"
        stored_context = session.get_context()
        
        response_text = generate(
            query="",
            context=f"참고 정보: {stored_context}\n\n---대화 기록---\n{chat_history_prompt}"
        )
        
        print(f'[Session {session.session_id}] 기존 컨텍스트로 응답 - 검색 횟수: {session.get_search_count()}')
        return response_text
    except Exception as e:
        print(f'[Session {session.session_id}] 응답 생성 오류: {e}')
        return f"{ui_messages.error_prefix}{str(e)}"


@handle_exceptions(default_return="")
def test_nlu_module(query: str) -> str:
    """NLU 모듈 테스트 - 의도 분류 및 엔티티 추출"""
    from ..retrieve.nlu import classify_intent_and_extract_entities
    result = classify_intent_and_extract_entities(query)
    return safe_format_json(result)


@handle_exceptions(default_return=("", "", ""))
def test_search_module(query: str) -> tuple[str, str, str]:
    """검색 모듈 전체 파이프라인 테스트"""
    from app.retrieve.nlu import classify_intent_and_extract_entities
    from app.retrieve.hybrid_search import (
        hybrid_search, 
        build_bm25_query, 
        build_vector_query
    )
    from app.retrieve.embeddings import get_query_embedding
    from app.retrieve.search import search_restaurants_by_intent, filter_by_relevance
    
    # 1. NLU 분석 (새로운 suggested_queries 포함)
    nlu_result = classify_intent_and_extract_entities(query)
    intent = nlu_result.get("intent", "search")
    entities = nlu_result.get("entities", {})
    suggested_queries = nlu_result.get("suggested_queries", [query])
    
    # 2. 검색 크기 결정
    if intent == "search":
        size = 5
    elif intent == "compare":
        size = 8
    elif intent == "information":
        size = 3
    else:
        size = 5
    
    # 3. Elasticsearch 쿼리 생성 (표시용 - 첫 번째 suggested_query 사용)
    display_query = suggested_queries[0] if suggested_queries else query
    search_size = max(50, size * 10)
    
    # BM25 쿼리 생성
    bm25_query = build_bm25_query(display_query, entities, search_size)
    
    # 벡터 쿼리 생성
    query_embedding = get_query_embedding([display_query])
    vector_query = build_vector_query(query_embedding, entities, search_size)
    
    # 벡터 요약으로 대체 (출력용)
    import copy
    display_vector_query = copy.deepcopy(vector_query)
    embedding_summary = f"[{len(query_embedding)}차원 벡터]" if query_embedding else "임베딩 없음"
    if "knn" in display_vector_query:
        display_vector_query["knn"]["query_vector"] = embedding_summary
    
    # 4. 연관도 필터링 이전 검색 결과 (하이브리드 검색만)
    pre_filter_results = hybrid_search(suggested_queries, entities, intent, size)
    
    # 5. 연관도 필터링 후 최종 검색 결과
    final_results = filter_by_relevance(query, pre_filter_results) if pre_filter_results else []
    
    # 출력 포맷팅
    nlu_str = safe_format_json(nlu_result)
    
    # Elasticsearch 쿼리 정보
    es_queries = {
        "BM25_쿼리": bm25_query,
        "벡터_쿼리": display_vector_query,
        "검색_크기": search_size,
        "검색_의도": intent,
        "사용된_쿼리들": suggested_queries
    }
    es_query_str = safe_format_json(es_queries)
    
    # 검색 결과 출력 (필터링 전/후 모두 표시)
    search_results_text = ""
    
    # 필터링 이전 결과
    search_results_text += "=== 연관도 필터링 이전 결과 ===\n\n"
    if pre_filter_results:
        for i, result in enumerate(pre_filter_results[:3], 1):  # 상위 3개만
            title = result.get("title", "제목 없음")
            summary = result.get("summary", "N/A")
            category = result.get("category", "카테고리 없음")
            convenience = result.get("convenience", [])
            
            # 하이브리드 검색 점수 정보 표시
            rrf_score = result.get("rrf_score", "N/A")
            search_method = result.get("search_method", "N/A")
            bm25_rank = result.get("bm25_rank", "N/A")
            vector_rank = result.get("vector_rank", "N/A")
            
            search_results_text += f"{i}. {title}\n"
            search_results_text += f"   RRF점수: {rrf_score}, 방법: {search_method}\n"
            if bm25_rank != "N/A":
                search_results_text += f"   BM25 순위: {bm25_rank}\n"
            if vector_rank != "N/A":
                search_results_text += f"   벡터 순위: {vector_rank}\n"
            search_results_text += f"   요약:\n{summary}...\n\n"
    else:
        search_results_text += "검색 결과가 없습니다.\n\n"
    
    # 필터링 이후 결과
    search_results_text += "=== 연관도 필터링 후 최종 결과 ===\n\n"
    if final_results:
        for i, result in enumerate(final_results[:3], 1):  # 상위 3개만
            title = result.get("title", "제목 없음")
            summary = result.get("summary", "N/A")
            category = result.get("category", "카테고리 없음")
            convenience = result.get("convenience", [])
            
            # 하이브리드 검색 점수 정보 표시
            rrf_score = result.get("rrf_score", "N/A")
            search_method = result.get("search_method", "N/A")
            bm25_rank = result.get("bm25_rank", "N/A")
            vector_rank = result.get("vector_rank", "N/A")
            
            search_results_text += f"{i}. {title}\n"
            search_results_text += f"   카테고리: {category}\n"
            search_results_text += f"   편의사항: {convenience}\n"
            search_results_text += f"   RRF점수: {rrf_score}, 방법: {search_method}\n"
            if bm25_rank != "N/A":
                search_results_text += f"   BM25 순위: {bm25_rank}\n"
            if vector_rank != "N/A":
                search_results_text += f"   벡터 순위: {vector_rank}\n"
            search_results_text += f"   요약: {summary[:200]}...\n\n"
    else:
        search_results_text += "연관도 필터링으로 인해 모든 결과가 제거되었습니다.\n\n"
    
    return (
        f"NLU 분석 결과:\n{nlu_str}\n\n",
        f"Elasticsearch 쿼리:\n{es_query_str}",
        f"검색 결과 비교:\n\n{search_results_text}"
    )


@handle_exceptions(default_return="")
def get_relevance_evaluation(query: str) -> str:
    """연관성 평가 결과만 반환"""
    # 실제 Elasticsearch 검색 수행
    documents = search_restaurants(query)
    
    if not documents:
        return ui_messages.no_search_results
    
    # 연관성 평가 수행
    relevance_result = grade_relevance(query, documents)
    
    # 결과 포맷팅
    result_text = f"=== 연관성 평가 결과 ===\n"
    result_text += f"쿼리: {query}\n"
    result_text += f"검색된 문서 수: {len(documents)}개\n\n"
    
    # 연관성 평가 결과 표시
    if relevance_result:
        overall_relevance = relevance_result.get("overall_relevance", "알 수 없음")
        overall_reason = relevance_result.get("reason", "이유 없음")
        document_scores = relevance_result.get("document_scores", [])
        
        result_text += f"전체 연관성: {overall_relevance}\n"
        result_text += f"전체 판단 근거: {overall_reason}\n\n"
        
        if document_scores:
            result_text += "개별 문서 연관성 평가:\n"
            for score in document_scores:
                doc_id = score.get("document_id", "알 수 없음")
                relevance = score.get("relevance", "알 수 없음")
                reason = score.get("reason", "이유 없음")
                
                # 해당 문서의 제목과 요약 가져오기
                try:
                    doc_index = int(doc_id) - 1
                    if 0 <= doc_index < len(documents):
                        doc = documents[doc_index]
                        title = doc.get("title", "제목 없음")
                    else:
                        title = "제목 없음"
                except (ValueError, TypeError):
                    title = "제목 없음"
                
                result_text += f"\n문서 {doc_id}: {title}\n"
                result_text += f"   연관성: {relevance}\n"
                result_text += f"   판단 근거: {reason}\n"
        else:
            result_text += "개별 문서 평가 결과가 없습니다.\n"
    else:
        result_text += "연관성 평가 결과를 가져올 수 없습니다.\n"
        
    return result_text


@handle_exceptions(default_return="")
def get_search_results_summary(query: str) -> str:
    """검색 결과 요약만 반환"""
    # 실제 Elasticsearch 검색 수행
    documents = search_restaurants(query)
    
    if not documents:
        return ui_messages.no_search_results
    
    # 검색 결과 요약 표시
    result_text = f"=== 검색 결과 요약 ===\n"
    result_text += f"총 {len(documents)}개 문서 검색됨\n\n"
    
    for i, doc in enumerate(documents, 1):
        title = doc.get("title", "제목 없음")
        summary = doc.get("summary", "요약 정보 없음")
        search_score = doc.get("_score", "점수 없음")
        result_text += f"{i}. {title}\n"
        result_text += f"   검색 점수: {search_score}\n"
        result_text += f"   요약: {summary}\n\n"
    
    return result_text


@handle_exceptions(default_return="")
def test_full_pipeline(query: str) -> str:
    """전체 파이프라인 테스트 (실제 검색 + 연관성 평가)"""
    # 실제 검색 수행
    results = search_restaurants(query)
    
    if not results:
        return ui_messages.no_search_results
    
    # 연관성 평가
    relevance_result = grade_relevance(query, results)
    
    # 결과 정리
    pipeline_result = {
        "검색_결과_수": len(results),
        "연관성_평가": relevance_result,
        "검색된_업체들": [
            {
                "업체명": doc.get("title", "N/A"),
                "점수": doc.get("_score", "N/A"),
                "요약": doc.get("summary", "N/A")[:100] + "..." 
                       if len(doc.get("summary", "")) > 100 
                       else doc.get("summary", "N/A")
            }
            for doc in results
        ]
    }
    
    return safe_format_json(pipeline_result)


def create_interface() -> gr.Blocks:
    """통합 Gradio 인터페이스 생성"""
    
    with gr.Blocks(title="맛집 검색 LLM 에이전트", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🍽️ 맛집 검색 LLM 에이전트")
        
        with gr.Tabs():
            # 채팅 인터페이스
            with gr.Tab("💬 채팅"):
                create_chat_interface(chat_fn)
            
            # 어드민 대시보드
            with gr.Tab("🔧 어드민 대시보드"):
                create_admin_dashboard(
                    test_nlu_module,
                    test_search_module,
                    get_relevance_evaluation,
                    get_search_results_summary
                )
    
    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name=config.server_name,
        server_port=config.server_port,
        share=config.share
    )