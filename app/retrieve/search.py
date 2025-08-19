"""
통합 식당 검색 모듈
"""

from typing import Any
from dotenv import load_dotenv
from tavily import TavilyClient

from .nlu import classify_intent_and_extract_entities
from .relevance import grade_relevance
from .hybrid_search import hybrid_search

load_dotenv()

tavily_client = TavilyClient()


def filter_by_relevance(query: str, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """연관도 기반 결과 필터링"""
    relevance_result = grade_relevance(query, results)
    print(f"연관도 평가: {relevance_result['overall_relevance']}")
    print(f"평가 근거: {relevance_result['reason']}")
    
    # relevant 문서만 필터링
    if relevance_result['overall_relevance'] == 'relevant':
        filtered_results = []
        document_scores = relevance_result.get('document_scores', [])
        
        for i, doc in enumerate(results):
            # document_scores에서 해당 문서의 relevance 확인
            doc_relevance = 'relevant'  # 기본값
            for score in document_scores:
                if str(score.get('document_id')) == str(i + 1):
                    doc_relevance = score.get('relevance', 'relevant')
                    break
            
            if doc_relevance == 'relevant':
                filtered_results.append(doc)
        
        print(f"필터링 후: {len(filtered_results)}개 문서\n")
        return filtered_results
    else:
        print("전체적으로 관련성이 낮은 것으로 판단되어 빈 결과를 반환합니다.")
        return []


# 메인 검색 함수들
def search_restaurants_by_intent(intent: str, entities: dict[str, Any], suggested_queries: list[str], original_query: str) -> list[dict[str, Any]]:
    """의도에 따른 하이브리드 검색 실행"""
    
    # 검색 전략에 따라 결과 수 조정
    if intent == "search":
        size = 5
    elif intent == "compare":
        size = 8  # 비교 대상이 여러 개일 수 있으므로 더 많이
    elif intent == "information":
        size = 3  # 특정 정보 요청이므로 적게
    else:
        print(f"지원하지 않는 검색 의도: {intent}")
        return []
    
    # 하이브리드 검색 실행 (suggested_queries 사용)
    results = hybrid_search(suggested_queries, entities, intent, size)
    
    print(f"{intent} 하이브리드 검색 완료: {len(results)}개 문서 발견")
    
    # 연관도 판단 및 필터링 (원본 쿼리로 평가)
    if results:
        return filter_by_relevance(original_query, results)
    
    return results


def search_restaurants(query: str) -> list[dict[str, Any]]:
    """
    자연어 쿼리를 받아서 식당 검색을 수행하는 메인 함수
    
    Args:
        query: 사용자의 자연어 쿼리

    Returns:
        검색된 식당 문서 리스트
    """
    
    # 1. 의도 분류, 엔티티 추출, 쿼리 재정의
    intent_result = classify_intent_and_extract_entities(query)
    intent = intent_result["intent"]
    entities = intent_result["entities"]
    suggested_queries = intent_result.get("suggested_queries", [query])
    
    print(f"쿼리: {query}")
    print(f"검색 의도: {intent}")
    print(f"추출된 엔티티: {entities}")
    print(f"재정의된 쿼리: {suggested_queries}")
    
    # 2. 의도에 따른 검색 전략 실행
    results = search_restaurants_by_intent(intent, entities, suggested_queries, query)
    
    return results


def search_web(query: str) -> list[dict[str, str]]:
    """웹 검색 실행"""
    response = tavily_client.search(
        query=query,
        max_results=3,
        search_depth='basic',
    )

    docs = [
        {
            "title": result["title"],
            "content": result["content"],
        } for result in response["results"]
    ]
    print(f"웹 검색 완료: {len(docs)}개 문서 발견")

    return docs 


def search(query: str) -> str:
    """통합 검색 (식당 검색 -> 웹 검색)"""
    context = ""
    docs = search_restaurants(query)
    
    if docs:
        for i, doc in enumerate(docs):
            context += f"""
문서 {i + 1}:
{doc["summary"]}

"""
    else:
        print("식당을 찾지 못해 웹 검색을 시작합니다.")
        docs = search_web(query)
        for i, doc in enumerate(docs):
            context += f"""
문서 {i + 1}:
문서 제목: {doc["title"]}
문서 내용: {doc["content"]}

"""
    return context

def test_search():
    """하이브리드 검색 통합 테스트"""
    test_queries = [
        "강남역 주차되는 일식집",  # search intent
        "마포 진대감 주차되나요?",  # information intent
        "강남 진대감 vs 굽다",  # compare intent
        "판교 가족외식 고기집 추천",  # search intent
        "데이트하기 좋은 로맨틱한 분위기 식당",  # search intent
    ]

    print("=== 하이브리드 검색 통합 테스트 ===")
    for query in test_queries:
        print(f"\n쿼리: {query}")
        print("-" * 60)
        context = search(query)
        # print(context)
        print("=" * 80)


if __name__ == "__main__":
    test_search()