"""
하이브리드 검색 모듈 (BM25 + 벡터 검색 + RRF 재순위화)
BM25 검색 시 엔티티를 활용한 부스팅을 적용하고, 벡터 검색은 location 필터만 사용
"""

import os
from typing import Any
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

from .embeddings import get_query_embedding

load_dotenv()


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


def build_location_filters(locations: list[str]) -> list[dict]:
    """location 엔티티로만 필터 생성"""
    filters = []
    
    # location 처리만 수행
    for location in locations:
        coordinates_results = search_coordinates_index(location)
        
        if coordinates_results:
            # 정확한 위치가 찾아진 경우
            if (len(coordinates_results) == 1) or (coordinates_results[0]["name"].strip() == location.strip()):
                coord = coordinates_results[0]["pin"]["coordinate"]
                filters.append({
                    "geo_distance": {
                        "distance": "3km",
                        "pin.coordinate": {
                            "lat": coord["lat"],
                            "lon": coord["lon"]
                        }
                    }
                })
            else:
                # 여러 결과가 있는 경우 주소 매칭으로 대체
                filters.append({
                    "match": {
                        "address": {
                            "query": location
                        }
                    }
                })
    
    return filters


def build_bm25_query(
    intent: str,
    query_text: str,
    entities: dict[str, Any],
    size: int = 50
) -> dict[str, Any]:
    """의도에 따른 BM25 키워드 검색 쿼리 생성"""
    
    should_clauses = []
    must_clauses = []

    search_fields = [
        "title^2.0", "category^1.5", "review_food^1.5", "convenience^1.2",
        "atmosphere^1.2", "occasion^1.2", "features^1.2", "address^0.8"
    ]

    should_clauses.append({
        "multi_match": {
            "query": query_text,
            "fields": search_fields,
            "type": "cross_fields",
            "boost": 0.5
        }
    })

    if titles := entities.get("title"):
        should_clauses.extend([{"match": {"title": {"query": title, "boost": 1.0}}} for title in titles])

    if categories := entities.get("category"):
        should_clauses.append({"match": {"category": {"query": ",".join(categories), "boost": 2.0}}})

    if conveniences := entities.get("convenience"):
        if intent == "search":
            must_clauses.append({"match": {"convenience": {"query": ",".join(conveniences), "boost": 1.2}}})
        else:
            should_clauses.append({"match": {"convenience": {"query": ",".join(conveniences), "boost": 1.2}}})

    if atmospheres := entities.get("atmosphere"):
        should_clauses.append({"match": {"atmosphere": {"query": ",".join(atmospheres), "boost": 1.0}}})

    if occasions := entities.get("occasion"):
        should_clauses.append({"match": {"occasion": {"query": ",".join(occasions), "boost": 1.0}}})

    if menus := entities.get("menu"):
        should_clauses.append({
            "nested": {
                "path": "menus",
                "query": {"match": {"review_food": {"query": ",".join(menus), "boost": 1.2}}},
            }
        })
        should_clauses.append({"match": {"review_food": {"query": ",".join(menus), "boost": 1.2}}})

    if locations := entities.get("location"):
        location_filters = build_location_filters(locations)
        if location_filters:
            must_clauses.extend(location_filters)

    final_bool_query = {}

    if should_clauses:
        final_bool_query["should"] = should_clauses
        final_bool_query["minimum_should_match"] = 1

    if must_clauses:
        final_bool_query["must"] = must_clauses

    query = {"bool": final_bool_query}

    return {
        "size": size,
        "_source": {
            "includes": [
                "place_id", "title", "summary", "category", "address",
                "convenience", "atmosphere", "occasion", "pin",
            ]
        },
        "query": query
    }


def build_vector_query(
    query_embedding: list[float],
    entities: dict[str, Any],
    size: int = 50
) -> dict[str, Any]:
    """벡터 검색 쿼리 생성 (location 필터만 적용)"""
    
    base_query = {
        "size": size,
        "_source": {
            "includes": ["place_id", "title", "summary", "category", "address", "convenience", "atmosphere", "occasion", "pin"]
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": size,
            "num_candidates": size * 4,  # 후보군을 충분히 크게
        }
    }
    
    # location 필터만 추가
    if locations := entities.get("location"):
        location_filters = build_location_filters(locations)
        if location_filters:
            base_query["knn"]["filter"] = location_filters
    
    return base_query


def execute_bm25_search(query_text: str, entities: dict[str, Any], intent: str = "search", size: int = 50) -> list[dict[str, Any]]:
    """BM25 검색 실행"""
    bm25_query = build_bm25_query(intent, query_text, entities, size)
    
    try:
        client = get_elasticsearch_client()
        response = client.search(index="restaurants", body=bm25_query)
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            doc["bm25_rank"] = len(results) + 1  # BM25 순위
            results.append(doc)
        
        return results
        
    except Exception as e:
        print(f"BM25 검색 오류: {e}")
        return []


def execute_vector_search(query_text: str, entities: dict[str, Any], intent: str = "search", size: int = 50) -> list[dict[str, Any]]:
    """벡터 검색 실행"""
    # 쿼리 임베딩 생성
    query_embedding = get_query_embedding([query_text])
    
    vector_query = build_vector_query(query_embedding, entities, size)
    
    try:
        client = get_elasticsearch_client()
        response = client.search(index="restaurants", body=vector_query)
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            doc["_rank"] = len(results) + 1  # 벡터 순위
            results.append(doc)
        
        return results
        
    except Exception as e:
        print(f"벡터 검색 오류: {e}")
        return []


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]], 
    vector_results: list[dict[str, Any]], 
    k: int = 60
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion으로 두 검색 결과를 결합
    
    Args:
        bm25_results: BM25 검색 결과
        vector_results: 벡터 검색 결과
        k: RRF 파라미터 (일반적으로 60 사용)
    
    Returns:
        RRF 점수로 재순위화된 결과
    """
    
    # place_id별로 점수를 집계
    rrf_scores = {}
    all_docs = {}
    
    # BM25 결과 처리
    for rank, doc in enumerate(bm25_results, 1):
        place_id = doc["place_id"]
        rrf_score = 1 / (k + rank)
        
        if place_id not in rrf_scores:
            rrf_scores[place_id] = 0
            all_docs[place_id] = doc
        
        rrf_scores[place_id] += rrf_score
        all_docs[place_id]["bm25_rank"] = rank
        all_docs[place_id]["bm25_score"] = doc["_score"]
    
    # 벡터 검색 결과 처리
    for rank, doc in enumerate(vector_results, 1):
        place_id = doc["place_id"]
        rrf_score = 1 / (k + rank)
        
        if place_id not in rrf_scores:
            rrf_scores[place_id] = 0
            all_docs[place_id] = doc
            all_docs[place_id]["bm25_rank"] = None
            all_docs[place_id]["bm25_score"] = None
        
        rrf_scores[place_id] += rrf_score
        all_docs[place_id]["vector_rank"] = rank
        all_docs[place_id]["vector_score"] = doc["_score"]
    
    # RRF 점수로 정렬
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 최종 결과 생성
    final_results = []
    for place_id, rrf_score in sorted_items:
        doc = all_docs[place_id].copy()
        doc["rrf_score"] = rrf_score
        doc["final_rank"] = len(final_results) + 1
        
        # 검색 방법 표시
        if "bm25_rank" in doc and doc.get("vector_rank") is not None:
            doc["search_method"] = "hybrid"
        elif "bm25_rank" in doc:
            doc["search_method"] = "bm25_only"
        else:
            doc["search_method"] = "vector_only"
        
        final_results.append(doc)
    
    return final_results


def hybrid_search(queries: list[str], entities: dict[str, Any], intent: str = "search", size: int = 5) -> list[dict[str, Any]]:
    """
    의도에 따른 하이브리드 검색 실행 (BM25 + 벡터 + RRF)
    
    Args:
        queries: 검색 쿼리 리스트 (NLU의 suggested_queries)
        entities: 추출된 엔티티
        intent: 검색 의도 (search, compare, information)
        size: 최종 반환할 결과 수
    
    Returns:
        RRF로 재순위화된 검색 결과
    """
    if not queries:
        print("검색 쿼리가 없습니다.")
        return []

    query_results = []
    
    # 각 쿼리별로 개별 검색 + RRF 실행
    for query in queries:
        # 쿼리 수에 따라 검색 사이즈 동적 조정
        if len(queries) > 1:
            # 다중 쿼리 시 각 쿼리별 보장 결과 수의 4배, 최소 20개
            guaranteed_results = max(1, size // len(queries))
            search_size = max(20, guaranteed_results * 4)
        else:
            search_size = max(50, size * 10) # 단일 쿼리 시 넉넉하게

        print(f"'{query}' BM25 검색 실행 중... (상위 {search_size}개)")
        bm25_results = execute_bm25_search(query, entities, intent, search_size)
        
        print(f"'{query}' 벡터 검색 실행 중... (상위 {search_size}개)")
        vector_results = execute_vector_search(query, entities, intent, search_size)
        
        if bm25_results or vector_results:
            query_rrf_results = reciprocal_rank_fusion(bm25_results, vector_results)
            query_results.append({
                'query': query,
                'results': query_rrf_results
            })
            print(f"'{query}' RRF 완료: {len(query_rrf_results)}개")
        else:
            print(f"'{query}' 검색 결과 없음")

    if not query_results:
        print("모든 쿼리에서 검색 결과가 없습니다.")
        return []

    # 결과 통합
    final_results = []
    seen_ids = set()

    if len(query_results) == 1:
        final_results = query_results[0]['results']
    else:
        # 여러 쿼리 결과 균등하게 섞기
        results_per_query = max(1, size // len(query_results))
        print(f"쿼리별 균등 분배: 각 쿼리당 {results_per_query}개씩")

        # 각 쿼리에서 최소 개수만큼 선택
        for query_data in query_results:
            query_name = query_data['query']
            added_count = 0
            for result in query_data['results']:
                if added_count >= results_per_query:
                    break
                if result['place_id'] not in seen_ids:
                    final_results.append(result)
                    seen_ids.add(result['place_id'])
                    added_count += 1
            print(f"'{query_name}': {added_count}개 선택")
        
        # 남은 슬롯을 상위 결과로 채우기 (쿼리 순서대로)
        if len(final_results) < size:
            for query_data in query_results:
                if len(final_results) >= size:
                    break
                for result in query_data['results']:
                    if len(final_results) >= size:
                        break
                    if result['place_id'] not in seen_ids:
                        final_results.append(result)
                        seen_ids.add(result['place_id'])

    # 최종 결과 반환
    final_results = final_results[:size]
    
    print(f"{intent} 하이브리드 검색 완료: {len(final_results)}개 문서 (RRF 재순위화)")
    
    # 각 검색 방법별 결과 수 출력
    if final_results:
        hybrid_count = sum(1 for doc in final_results if doc.get("search_method") == "hybrid")
        bm25_only_count = sum(1 for doc in final_results if doc.get("search_method") == "bm25_only")
        vector_only_count = sum(1 for doc in final_results if doc.get("search_method") == "vector_only")
        print(f"결과 분석: 하이브리드={hybrid_count}, BM25만={bm25_only_count}, 벡터만={vector_only_count}")
    
    return final_results


def test_hybrid_search():
    """하이브리드 검색 테스트"""
    
    test_cases = [
        {
            "query": "강남역 주차되는 일식집",
            "entities": {
                "location": ["강남역"],
                "category": ["일식"],
                "convenience": ["주차"]
            }
        },
        {
            "query": "판교 애견동반 식당",
            "entities": {
                "location": ["판교"],
                "convenience": ["반려동물"]
            }
        },
        {
            "query": "데이트하기 좋은 로맨틱한 분위기 식당",
            "entities": {
                "occasion": ["데이트"],
                "atmosphere": ["로맨틱한"]
            }
        },
        {
            "query": "버거킹 vs 맥도날드 메뉴 비교해줘",
            "entities": {
                "title": ["버거킹", "맥도날드"],
            }
        }
    ]
    
    print("=== 하이브리드 검색 테스트 ===")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 테스트 쿼리: '{test_case['query']}'")
        print(f"   엔티티: {test_case['entities']}")
        print(f"   적용 로직: location 필터링 + BM25 엔티티 부스팅")
        print("-" * 60)
        
        results = hybrid_search([test_case["query"]], test_case["entities"], size=5)
        
        for j, doc in enumerate(results, 1):
            print(f"{j}. {doc.get('title', 'N/A')} (RRF: {doc['rrf_score']:.4f}, 방법: {doc['search_method']})")
            print(f"   카테고리: {doc.get('category', 'N/A')}")
            if doc.get('bm25_rank'):
                print(f"   BM25 순위: {doc['bm25_rank']} (점수: {doc.get('bm25_score', 'N/A'):.2f})")
            if doc.get('vector_rank'):
                print(f"   벡터 순위: {doc['vector_rank']} (점수: {doc.get('vector_score', 'N/A'):.2f})")
            print(f"   주소: {doc.get('address', 'N/A')}")
            print(f"   편의: {doc.get('convenience', [])}")
            print(f"   분위기: {doc.get('atmosphere', [])}")
            print(f"   상황: {doc.get('occasion', [])}")
        
        print()


if __name__ == "__main__":
    test_hybrid_search()
