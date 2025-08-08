"""
Elasticsearch 관련 기능 모듈
"""

import os
from typing import Any
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

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
    if locations := structured_query.get("location"):
        for location in locations:
            if location["relation"] == "exact":
                # 특정 식당명으로 검색
                es_query["knn"]["filter"].append({
                    "match": {
                        "title": {
                            "query": location["name"],
                        }
                    }
                })
            elif location["relation"] == "nearby":
                # 위치 기반 검색
                # poi_location = search_naver_poi(location["name"])
                # if poi_location:
                #     es_query["knn"]["filter"].append({
                #         "geo_distance": {
                #             "distance": "3km",
                #             "pin.coordinate": {
                #                 "lat": poi_location["lat"],
                #                 "lon": poi_location["lon"]
                #             }
                #         }
                #     })
                pass
    
    # convenience 필터링 (필수 조건)
    if conveniences := structured_query.get("convenience"):
        conveniences_text = ",".join(conveniences)
        es_query["knn"]["filter"].append({
            "match": {
                "convenience": {
                    "query": conveniences_text,
                    "operator": "and",
                }
            }
        })
    
    # category, menu 필터링 (음식 관련)
    if category := structured_query.get("category"):
        es_query["knn"]["filter"].append(
            {
                "match": {
                    "category": {
                        "query": category,
                        "operator": "and"
                    },
                }
            }
        )
    
    if menus := structured_query.get("menu"):
        menus_text = ",".join(menus)
        es_query["knn"]["filter"].append(
            {
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
            }
        )

    return es_query


def search_elasticsearch(es_query: dict[str, Any], index_name: str = "restaurants") -> list[dict[str, Any]]:
    """Elasticsearch에서 검색 실행"""
    es = create_elasticsearch_client()
    try:
        response = es.search(index=index_name, body=es_query)
        
        results = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["_score"] = hit["_score"]
            results.append(doc)
        
        return results
        
    except Exception as e:
        print(f"Elasticsearch 검색 오류: {e}")
        return []