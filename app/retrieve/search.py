"""
통합 식당 검색 모듈
"""

import os
from typing import Any
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from tavily import TavilyClient

from .nlu import classify_intent_and_extract_entities
from .embeddings import get_query_embedding
from .relevance import grade_relevance

load_dotenv()

tavily_client = TavilyClient()


# Elasticsearch 클라이언트 생성
def create_elasticsearch_client() -> Elasticsearch:
    """Elasticsearch 클라이언트 생성"""
    host = os.environ.get("ELASTICSEARCH_HOST")
    username = os.environ.get("ELASTICSEARCH_USERNAME")
    password = os.environ.get("ELASTICSEARCH_PASSWORD")

    return Elasticsearch(
        [host],
        basic_auth=(username, password),
        verify_certs=False
    )


# 전역 클라이언트 인스턴스
_es_client = None


def get_elasticsearch_client() -> Elasticsearch:
    """Elasticsearch 클라이언트 반환 (싱글톤 패턴)"""
    global _es_client
    if _es_client is None:
        _es_client = create_elasticsearch_client()
    return _es_client


# 기본 쿼리 생성 함수들
def build_base_query(
    size: int, 
    query_embedding: list[float], 
    k: int = None, 
    num_candidates: int = 100
) -> dict[str, Any]:
    """기본 벡터 검색 쿼리 생성"""
    if k is None:
        k = size
        
    return {
        "size": size,
        "_source": {
            "includes": ["place_id", "summary"],
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": k,
            "num_candidates": num_candidates,
            "filter": [],
        }
    }


def add_location_filter(
    es_query: dict[str, Any], 
    location: str, 
    relation: str = "nearby"
) -> None:
    """위치 필터 추가"""
    if relation == "exact":
        # 특정 식당명으로 검색
        es_query["knn"]["filter"].append({
            "match": {
                "title": {
                    "query": location,
                }
            }
        })
    elif relation == "nearby":
        coordinates_results = search_coordinates_index(location)
        
        if coordinates_results:
            if (len(coordinates_results) == 1) or (coordinates_results[0]["name"].strip() == location.strip()):
                coord = coordinates_results[0]["pin"]["coordinate"]
                es_query["knn"]["filter"].append({
                    "geo_distance": {
                        "distance": "3km",
                        "pin.coordinate": {
                            "lat": coord["lat"],
                            "lon": coord["lon"]
                        }
                    }
                })
            else:
                es_query["knn"]["filter"].append({
                    "match": {
                        "address": {
                            "query": location
                        }
                    }
                })


def add_title_filter(es_query: dict[str, Any], title: str) -> None:
    """식당명 필터 추가"""
    es_query["knn"]["filter"].append({
        "match": {
            "title": {
                "query": title,
                "operator": "and"
            }
        }
    })


def add_convenience_filter(es_query: dict[str, Any], conveniences: list[str]) -> None:
    """편의시설 필터 추가"""
    conveniences_text = ",".join(conveniences)
    es_query["knn"]["filter"].append({
        "match": {
            "convenience": {
                "query": conveniences_text,
                "operator": "and",
            }
        }
    })


def add_category_filter(es_query: dict[str, Any], category: str) -> None:
    """카테고리 필터 추가"""
    es_query["knn"]["filter"].append({
        "match": {
            "category": {
                "query": category,
                "operator": "and"
            },
        }
    })


def add_menu_filter(es_query: dict[str, Any], menus: list[str]) -> None:
    """메뉴 필터 추가"""
    menus_text = ",".join(menus)
    es_query["knn"]["filter"].append({
        "bool": {
            "should": [
                {
                    "nested": {
                        "path": "menus",
                        "query": {
                            "match": {
                                "menus.name": {
                                    "query": menus_text,
                                    "operator": "or",
                                },
                            },
                        },
                    },
                },
                {
                    "match": {
                        "review_food": {
                            "query": menus_text,
                            "operator": "or",
                        },
                    },
                },
            ],
            "minimum_should_match": 1,
        }
    })


# Elasticsearch 검색 함수들
def search_coordinates_index(query: str) -> list[dict[str, Any]]:
    """coordinates 인덱스에서 검색하는 함수"""
    search_query = {
        "query": {
            "match": {
                "name": {
                    "query": query,
                    "operator": "and"
                }
            }
        },
    }
    
    try:
        client = get_elasticsearch_client()
        response = client.search(index="coordinates", body=search_query)
        results = []
        for hit in response["hits"]["hits"]:
            results.append(hit["_source"])
        
        return results
        
    except Exception as e:
        print(f"coordinates 인덱스 검색 오류: {e}")
        return []


def elasticsearch_search(es_query: dict[str, Any]) -> list[dict[str, Any]]:
    """Elasticsearch에서 검색 실행"""
    try:
        client = get_elasticsearch_client()
        response = client.search(index="restaurants", body=es_query)
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            results.append(doc)
        
        return results
        
    except Exception as e:
        print(f"Elasticsearch 검색 오류: {e}")
        return []


# 검색 전략 함수들
def build_general_search_query(entities: dict[str, Any], query_embedding: list[float]) -> dict[str, Any]:
    """일반 검색용 Elasticsearch 쿼리 생성"""
    es_query = build_base_query(3, query_embedding, 3)
    
    # location 처리
    if locations := entities.get("location"):
        for location in locations:
            add_location_filter(es_query, location, "nearby")
    
    # title 처리 (식당명)
    if titles := entities.get("title"):
        for title in titles:
            add_title_filter(es_query, title)
    
    # convenience 필터링
    if conveniences := entities.get("convenience"):
        add_convenience_filter(es_query, conveniences)
    
    # category 필터링
    if categories := entities.get("category"):
        for category in categories:
            add_category_filter(es_query, category)
    
    # menu 필터링
    if menus := entities.get("menu"):
        add_menu_filter(es_query, menus)

    return es_query


def build_compare_search_queries(entities: dict[str, Any], query_embedding: list[float]) -> list[dict[str, Any]]:
    """비교 검색용 Elasticsearch 쿼리 생성 (각 식당별로 개별 쿼리)"""
    queries = []
    titles = entities.get("title", [])
    
    for title in titles:
        es_query = build_base_query(2, query_embedding, 2, 50)
        
        # 특정 식당명으로 필터링
        add_title_filter(es_query, title)
        
        # location 추가 필터링 (있는 경우)
        if locations := entities.get("location"):
            for location in locations:
                if location.lower() != title.lower():  # title과 다른 location만 필터로 추가
                    add_location_filter(es_query, location, "nearby")
        
        queries.append(es_query)
    
    return queries


def build_information_search_query(entities: dict[str, Any], query_embedding: list[float]) -> dict[str, Any]:
    """정보 요청용 Elasticsearch 쿼리 생성"""
    es_query = build_base_query(5, query_embedding, 5)
    
    # title 우선 처리 (특정 식당 정보 요청)
    if titles := entities.get("title"):
        for title in titles:
            add_title_filter(es_query, title)
    
    # location 처리 (title이 없는 경우에만)
    if locations := entities.get("location"):
        for location in locations:
            if not entities.get("title"):
                add_location_filter(es_query, location, "nearby")
    
    return es_query


def execute_general_search(entities: dict[str, Any], query_embedding: list[float]) -> list[dict[str, Any]]:
    """일반 검색 실행"""
    query = build_general_search_query(entities, query_embedding)
    return elasticsearch_search(query)


def execute_compare_search(entities: dict[str, Any], query_embedding: list[float]) -> list[dict[str, Any]]:
    """비교 검색 실행"""
    queries = build_compare_search_queries(entities, query_embedding)
    all_results = []
    
    for i, es_query in enumerate(queries):
        query_results = elasticsearch_search(es_query)
        print(f"비교 대상 {i+1} 검색: {len(query_results)}개 문서 발견")
        all_results.extend(query_results)
    
    # 중복 제거 (place_id 기준)
    seen_ids = set()
    results = []
    for doc in all_results:
        if doc["place_id"] not in seen_ids:
            results.append(doc)
            seen_ids.add(doc["place_id"])
    
    return results


def execute_information_search(entities: dict[str, Any], query_embedding: list[float]) -> list[dict[str, Any]]:
    """정보 요청 검색 실행"""
    query = build_information_search_query(entities, query_embedding)
    return elasticsearch_search(query)


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
def search_restaurants_by_intent(intent: str, entities: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """의도에 따른 검색 실행"""
    # 쿼리 임베딩 생성
    query_embedding = get_query_embedding([query])
    print(f"임베딩 생성 완료 (차원: {len(query_embedding)})")
    
    # 검색 전략에 따라 실행
    if intent == "search":
        results = execute_general_search(entities, query_embedding)
    elif intent == "compare":
        results = execute_compare_search(entities, query_embedding)
    elif intent == "information":
        results = execute_information_search(entities, query_embedding)
    else:
        print(f"지원하지 않는 검색 의도: {intent}")
        return []
    
    print(f"{intent} 검색 완료: {len(results)}개 문서 발견")
    
    # 연관도 판단 및 필터링
    if results:
        return filter_by_relevance(query, results)
    
    return results


def search_restaurants(query: str) -> list[dict[str, Any]]:
    """
    자연어 쿼리를 받아서 식당 검색을 수행하는 메인 함수
    
    Args:
        query: 사용자의 자연어 쿼리

    Returns:
        검색된 식당 문서 리스트
    """
    
    # 1. 의도 분류 및 엔티티 추출
    intent_result = classify_intent_and_extract_entities(query)
    intent = intent_result["intent"]
    entities = intent_result["entities"]
    
    print(f"쿼리: {query}")
    print(f"검색 의도: {intent}")
    print(f"추출된 엔티티: {entities}")
    
    # 2. 의도에 따른 검색 전략 실행
    results = search_restaurants_by_intent(intent, entities, query)
    
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
    """통합 검색 테스트"""
    test_queries = [
        "강남역 주차되는 일식집",  # search intent
        "마포 진대감 주차되나요?",  # information intent
        "강남 진대감 vs 굽다",  # compare intent
        "판교 가족외식 고기집 추천",  # search intent
    ]

    for query in test_queries:
        context = search(query)
        # print(context)
        print("=" * 80)


if __name__ == "__main__":
    test_search()