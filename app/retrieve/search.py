"""
자연어 쿼리를 받아서 쿼리 재구조화, 임베딩 계산, 검색을 수행하는 메인 모듈
"""

import json
import os
import requests
from typing import Any
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
import copy

from .query_rewrite import rewrite_query
from .embeddings import get_query_embedding
from .relevance import grade_relevance

load_dotenv()


def create_elasticsearch_client() -> Elasticsearch:
    """Elasticsearch 클라이언트 생성"""
    host = os.environ.get("ELASTICSEARCH_HOST")
    username = os.environ.get("ELASTICSEARCH_USERNAME")
    password = os.environ.get("ELASTICSEARCH_PASSWORD")

    return Elasticsearch(
        [f"http://{host}"],
        basic_auth=(username, password),
        verify_certs=False
    )


def search_naver_poi(query: str) -> dict[str, float] | None:
    """네이버 검색 API를 사용해서 POI 위치 정보 검색"""
    client_id = os.environ.get("NAVER_CLIENT_ID")
    client_secret = os.environ.get("NAVER_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Warning: Naver API credentials not found")
        return None
    
    url = "https://openapi.naver.com/v1/search/local.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    params = {
        "query": query,
        "display": 1,
        "start": 1,
        "sort": "random",
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data["items"]:
            item = data["items"][0]

            lon = float(f"{item['mapx'][:3]}.{item['mapx'][3:]}")
            lat = float(f"{item['mapy'][:2]}.{item['mapy'][2:]}")

            return {"lat": lat, "lon": lon}
            
    except Exception as e:
        print(f"Naver API error: {e}")
    
    return None


def build_elasticsearch_query(
    structured_query: dict[str, Any], 
    query_embedding: list[float]
) -> dict[str, Any]:
    """구조화된 쿼리를 바탕으로 Elasticsearch 쿼리 생성"""
    
    es_query = {
        "size": 3,
        "_source": {
            "includes": ["place_id", "summary"],
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_embedding,
            "k": 5,
            "num_candidates": 100,
            "filter": [],
        }
    }
    
    # location 처리
    if "location" in structured_query:
        for location in structured_query["location"]:
            if location["relation"] == "exact":
                # 특정 식당명으로 검색
                es_query["knn"]["filter"].append({
                    "match": {
                        "title": {
                            "query": location["name"],
                            "operator": "and",
                            "boost": 3.0
                        }
                    }
                })
            elif location["relation"] == "nearby":
                # 위치 기반 검색
                poi_location = search_naver_poi(location["name"])
                if poi_location:
                    es_query["knn"]["filter"].append({
                        "geo_distance": {
                            "distance": "30km",
                            "pin.location": {
                                "lat": poi_location["lat"],
                                "lon": poi_location["lon"]
                            }
                        }
                    })
    
    # convenience 필터링 (필수 조건)
    if "convenience" in structured_query:
        for convenience in structured_query["convenience"]:
            es_query["knn"]["filter"].append({
                "term": {
                    "convenience": convenience
                }
            })
    
    # cuisine, menu 필터링 (음식 관련)
    food_should = []
    if "cuisine" in structured_query:
        for cuisine in structured_query["cuisine"]:
            food_should.extend([
                {
                    "nested": {
                        "path": "menus",
                        "query": {
                            "match": {
                                "menus.name": {
                                    "query": cuisine,
                                    "operator": "and",
                                }
                            }
                        },
                    },
                },
                {
                    "match": {
                        "review_food": {
                            "query": cuisine,
                            "operator": "and",
                        },
                    },
                }
            ])
    
    if "menu" in structured_query:
        for menu in structured_query["menu"]:
            food_should.extend([
                {
                    "nested": {
                        "path": "menus",
                        "query": {
                            "match": {
                                "menus.name": {
                                    "query": menu,
                                    "operator": "and",
                                },
                            },
                        },
                    },
                },
                {
                    "match": {
                        "review_food": {
                            "query": menu,
                            "operator": "and",
                        },
                    },
                },
            ])
    
    if food_should:
        es_query["knn"]["filter"].append({
            "bool": {
                "should": food_should,
                "minimum_should_match": 1
            }
        })
    
    return es_query


def search_restaurants(query: str, index_name: str = "restaurants") -> list[dict[str, Any]]:
    """
    자연어 쿼리를 받아서 식당 검색을 수행하는 메인 함수
    
    Args:
        query: 사용자의 자연어 쿼리
        index_name: Elasticsearch 인덱스명
        
    Returns:
        검색된 식당 문서 리스트
    """
    
    # 1. 쿼리 재구조화
    structured_query = rewrite_query(query)
    print(f"구조화된 쿼리: {structured_query}")
    
    # 2. 쿼리 임베딩 생성
    query_embedding = get_query_embedding([query])
    print(f"임베딩 생성 완료 (차원: {len(query_embedding)})")
    
    # 3. Elasticsearch 쿼리 생성
    es_query = build_elasticsearch_query(structured_query, query_embedding)
    # display_es_query = copy.deepcopy(es_query)
    # display_es_query["knn"].pop("query_vector")
    # print(f"Elasticsearch 쿼리: {json.dumps(display_es_query, indent=2, ensure_ascii=False)}")
    
    # 4. 검색 실행
    es = create_elasticsearch_client()
    try:
        response = es.search(index=index_name, body=es_query)
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            results.append(doc)
        
        print(f"검색 완료: {len(results)}개 문서 발견")

        # 5. 연관도 판단 및 문서 필터링
        if results:
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

        return results
        
    except Exception as e:
        print(f"Elasticsearch 검색 오류: {e}")
        return []


def test_search():
    """검색 기능 테스트"""
    test_queries = [
        "강남역 주차되는 일식집",
        "판교 애견동반 식당",
        "마포 진대감 주차되나요?",
        "조용히 대화할 수 있는 맥주집",
        "송파 회식하기 좋은 삼겹살집 추천해줘"
    ]
    
    print("=== 식당 검색 테스트 ===")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 테스트 쿼리: '{query}'")
        print("=" * 50)
        
        results = search_restaurants(query)

        if results:
            for j, result in enumerate(results, 1):
                print(f"업체 {j}:\n")
                print(f"업체 ID: {result['place_id']}\n")
                print(f"업체 요약:\n{result['summary']}")
                print(f"점수: {result.get('_score')}")
                print('-' * 80)
        else:
            print("  검색 결과 없음")
        
        print()


if __name__ == "__main__":
    test_search()