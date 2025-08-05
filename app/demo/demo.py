"""
맛집 검색 LLM 에이전트의 통합 인터페이스
사용자용 채팅 인터페이스와 어드민 대시보드를 하나로 통합한 Gradio 앱
"""

import json
import gradio as gr
from dotenv import load_dotenv
from typing import Iterator
from ..retrieve.search import search, search_restaurants, build_elasticsearch_query, create_elasticsearch_client
from ..generation.generation import generate, generate_streaming
from ..retrieve.query_rewrite import rewrite_query
from ..retrieve.embeddings import get_query_embedding
from ..retrieve.relevance import grade_relevance

load_dotenv()

# 대화 상태를 저장할 전역 변수
conversation_context = None

def chat_fn(message: str, history: list[list[str]]) -> str:
    """
    채팅 처리 함수
    
    Args:
        message: 사용자의 현재 메시지
        history: 이전 대화 기록 [[user_msg, bot_response], ...]
        
    Returns:
        str: 응답 메시지
    """
    global conversation_context

    if not history:
        # 첫 대화: 검색을 수행하고 결과를 context에 저장
        query = message
        restaurant_context = search(query)
        conversation_context = restaurant_context  # 검색 결과를 전역 컨텍스트에 저장

        bot_response = generate(query, restaurant_context)
        print(f'bot_response: {bot_response}')
        return bot_response
    else:
        # 이어지는 대화: 저장된 context와 전체 대화 기록을 함께 사용
        # 대화 히스토리를 LLM에 제공할 프롬프트 형식으로 변환
        chat_history_prompt = ""
        for user_msg, bot_msg in history:
            chat_history_prompt += f"사용자: {user_msg}\n"
            chat_history_prompt += f"어시스턴트: {bot_msg}\n"
        
        # 현재 사용자 메시지 추가
        chat_history_prompt += f"사용자: {message}\n"

        # 저장된 레스토랑 정보와 대화 기록을 모두 컨텍스트로 활용
        response_text = generate(
            query="",
            context=f"참고 정보: {conversation_context}\n\n---대화 기록---\n{chat_history_prompt}"
        )
        return response_text


def test_query_rewrite(query: str) -> str:
    """쿼리 재작성 모듈 테스트"""
    try:
        result = rewrite_query(query)
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"오류 발생: {str(e)}"


def test_search_module(query: str) -> tuple[str, str, str]:
    """검색 모듈 전체 파이프라인 테스트"""
    try:
        # 1. 쿼리 재작성
        structured_query = rewrite_query(query)
        
        # 2. 임베딩 생성
        query_embedding = get_query_embedding([query])
        
        # 3. Elasticsearch 쿼리 생성
        es_query = build_elasticsearch_query(structured_query, query_embedding)
        
        # 4. 검색 실행
        es_client = create_elasticsearch_client()
        response = es_client.search(index="restaurants", body=es_query)
        
        # 결과 정리
        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            results.append(doc)
        
        # 출력 포맷팅
        structured_query_str = json.dumps(structured_query, ensure_ascii=False, indent=2)
        
        # Elasticsearch 쿼리에서 벡터 제거 후 표시
        display_es_query = es_query.copy()
        if "knn" in display_es_query and "query_vector" in display_es_query["knn"]:
            display_es_query["knn"]["query_vector"] = f"[벡터 차원: {len(query_embedding)}]"
        es_query_str = json.dumps(display_es_query, ensure_ascii=False, indent=2)
        
        # 검색 결과 - 요약만 표시
        search_results_text = ""
        for i, result in enumerate(results, 1):
            summary = result.get("summary", "N/A")
            score = result.get("_score", "N/A")
            
            search_results_text += f"{i}. (점수: {score})\n"
            search_results_text += f"{summary}\n\n\n"
        
        return (
            f"구조화된 쿼리:\n{structured_query_str}\n\n",
            f"Elasticsearch 쿼리:\n{es_query_str}",
            f"검색 결과 ({len(results)}개):\n\n{search_results_text}"
        )
        
    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        return error_msg, error_msg, error_msg


def get_relevance_evaluation(query: str) -> str:
    """연관성 평가 결과만 반환"""
    try:
        # 실제 Elasticsearch 검색 수행
        documents = search_restaurants(query)
        
        if not documents:
            return "검색 결과가 없습니다."
        
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
        
    except Exception as e:
        return f"오류 발생: {str(e)}"


def get_search_results_summary(query: str) -> str:
    """검색 결과 요약만 반환"""
    try:
        # 실제 Elasticsearch 검색 수행
        documents = search_restaurants(query)
        
        if not documents:
            return "검색 결과가 없습니다."
        
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
        
    except Exception as e:
        return f"오류 발생: {str(e)}"


def test_full_pipeline(query: str) -> str:
    """전체 파이프라인 테스트 (실제 검색 + 연관성 평가)"""
    try:
        # 실제 검색 수행
        results = search_restaurants(query)
        
        if not results:
            return "검색 결과가 없습니다."
        
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
                    "요약": (doc.get("summary", "N/A")[:100] + "..." 
                           if len(doc.get("summary", "")) > 100 
                           else doc.get("summary", "N/A"))
                }
                for doc in results
            ]
        }
        
        return json.dumps(pipeline_result, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"오류 발생: {str(e)}"


def create_interface():
    """통합 Gradio 인터페이스 생성"""
    
    with gr.Blocks(title="맛집 검색 LLM 에이전트", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🍽️ 맛집 검색 LLM 에이전트")
        
        with gr.Tabs():
            # 사용자 채팅 인터페이스
            with gr.Tab("💬 채팅"):
                chat_interface = gr.ChatInterface(
                    fn=chat_fn,
                    examples=[
                        "강남역 근처 맛있는 한식당 추천해줘",
                        "데이트하기 좋은 이태리 레스토랑이 어디있을까?",
                        "홍대에서 혼밥하기 좋은 곳 알려줘",
                        "회식하기 좋은 고기집 추천해줘"
                    ],
                    cache_examples=False,
                    chatbot=gr.Chatbot(height="80vh")
                )
            
            # 어드민 대시보드
            with gr.Tab("🔧 어드민 대시보드"):
                gr.Markdown("## 각 모듈의 결과를 확인하고 디버깅할 수 있는 인터페이스입니다.")
                
                with gr.Tabs():
                    # 쿼리 재작성 모듈 테스트
                    with gr.Tab("🔄 쿼리 재작성"):
                        gr.Markdown("### 자연어 쿼리를 구조화된 JSON으로 변환하는 모듈을 테스트합니다.")
                        
                        with gr.Row():
                            with gr.Column():
                                query_input = gr.Textbox(
                                    label="테스트 쿼리",
                                    placeholder="예: 강남역 주차되는 일식집",
                                    value="강남역 주차되는 일식집"
                                )
                                rewrite_btn = gr.Button("쿼리 재작성 테스트", variant="primary")
                            
                            with gr.Column():
                                rewrite_output = gr.Code(
                                    label="구조화된 쿼리 결과",
                                    language="json",
                                    lines=15
                                )
                        
                        rewrite_btn.click(
                            fn=test_query_rewrite,
                            inputs=[query_input],
                            outputs=[rewrite_output]
                        )
                    
                    # 검색 모듈 테스트
                    with gr.Tab("🔍 검색"):
                        gr.Markdown("### 쿼리 재작성 → 임베딩 → Elasticsearch 검색 파이프라인을 테스트합니다.")
                        
                        with gr.Row():
                            with gr.Column():
                                search_query_input = gr.Textbox(
                                    label="테스트 쿼리",
                                    placeholder="예: 마포 진대감 주차되나요?",
                                    value="마포 진대감 주차되나요?"
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
                            fn=test_search_module,
                            inputs=[search_query_input],
                            outputs=[structured_query_output, es_query_output, search_results_output]
                        )
                    
                    # 연관성 평가 모듈 테스트
                    with gr.Tab("⚖️ 연관성 평가"):
                        gr.Markdown("### 실제 Elasticsearch 검색 결과에 대한 연관성 평가와 요약 정보를 표시합니다.")
                        
                        with gr.Row():
                            with gr.Column():
                                relevance_query_input = gr.Textbox(
                                    label="테스트 쿼리",
                                    placeholder="예: 강남역 주차되는 일식집",
                                    value="강남역 주차되는 일식집"
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
                            fn=get_relevance_evaluation,
                            inputs=[relevance_query_input],
                            outputs=[relevance_output]
                        )
                        
                        relevance_btn.click(
                            fn=get_search_results_summary,
                            inputs=[relevance_query_input],
                            outputs=[search_results_output]
                        )
                    
                gr.Markdown("---")
                gr.Markdown("💡 **사용법**: 각 탭에서 다른 쿼리를 입력해서 각 모듈의 동작을 확인할 수 있습니다.")
    
    return demo


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )