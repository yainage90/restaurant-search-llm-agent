from dotenv import load_dotenv
import os
import json
import glob
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import Any


load_dotenv()


def create_elasticsearch_client(
) -> Elasticsearch:
    """Elasticsearch 클라이언트 생성"""
    host = os.environ.get("ELASTICSEARCH_HOST")
    username = os.environ.get("ELASTICSEARCH_USERNAME")
    password = os.environ.get("ELASTICSEARCH_PASSWORD")

    return Elasticsearch(
        [f"http://{host}"],
        basic_auth=(username, password),
        verify_certs=False
    )


def create_index_mapping(es: Elasticsearch, index_name: str) -> None:
    """Elasticsearch 인덱스 및 매핑 생성"""
    mapping = {
        "mappings": {
            "properties": {
                "place_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {
                        "keyword": {"type": "keyword"}
                    }
                },
                "address": {"type": "text", "analyzer": "standard"},
                "roadAddress": {"type": "text", "analyzer": "standard"},
                "pin": {
                    "properties": {
                        "location": {
                            "type": "geo_point",
                        }
                    }
                },
                "menus": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "text", "analyzer": "standard"},
                        "price": {"type": "integer"}
                    }
                },
                "reviews": {"type": "text", "analyzer": "standard"},
                "description": {"type": "text", "analyzer": "standard"},
                "review_food": {"type": "keyword"},
                "convenience": {"type": "keyword"},
                "atmosphere": {"type": "keyword"},
                "occasion": {"type": "keyword"},
                "features": {"type": "keyword"},
                "summary": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": True,
                    "similarity": "dot_product",
                    "index_options": {
                        "type": "hnsw",
                    }
                }
            }
        },
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 1
            },
        }
    }
    
    # 인덱스가 이미 존재하면 삭제
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"기존 인덱스 '{index_name}' 삭제됨")
    
    # 새 인덱스 생성
    es.indices.create(index=index_name, body=mapping)
    print(f"인덱스 '{index_name}' 생성됨")


def preprocess_document(doc: dict[str, Any]) -> dict[str, Any]:
    """문서 전처리"""
    processed_doc = doc.copy()
    
    # 위치 정보 추가 (geo_point 형식)
    processed_doc["pin"] = {
        "location": doc["location"],
    }
    
    # 메뉴 가격을 정수로 변환 (이미 변환되어 있다면 그대로 유지)
    if "menus" in doc and isinstance(doc["menus"], list):
        for menu in processed_doc["menus"]:
            if isinstance(menu.get("price"), str):
                # "20,000원" -> 20000
                price_str = menu["price"].replace(",", "").replace("원", "")
                try:
                    menu["price"] = int(price_str)
                except ValueError:
                    menu["price"] = 0
    
    return processed_doc


def index_document(es: Elasticsearch, index_name: str, doc: dict[str, Any]) -> None:
    """단일 문서 색인"""
    processed_doc = preprocess_document(doc)
    place_id = processed_doc.get("place_id")
    
    es.index(
        index=index_name,
        id=place_id,
        body=processed_doc
    )


def bulk_index_documents(es: Elasticsearch, index_name: str, documents: list[dict[str, Any]]) -> None:
    """배치로 문서들 색인"""
    actions = []
    
    for doc in documents:
        processed_doc = preprocess_document(doc)
        place_id = processed_doc.get("place_id")

        action = {
            "_index": index_name,
            "_id": place_id,
            "_source": processed_doc,
        }
        actions.append(action)
    
    # 배치 크기별로 나누어 색인
    batch_size = 100
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i + batch_size]
        bulk(es, batch)
        print(f"배치 {i//batch_size + 1} 색인 완료 ({len(batch)}개 문서)")


def load_and_index_from_json(
    documents_dir: str, 
    index_name: str = "restaurants",
) -> None:
    """documents 디렉토리의 part-*.jsonl 파일들에서 데이터를 읽어 Elasticsearch에 색인"""
    
    # Elasticsearch 클라이언트 생성
    es = create_elasticsearch_client()
    
    # 인덱스 생성
    create_index_mapping(es, index_name)
    
    # part-*.jsonl 파일들 찾기
    pattern = os.path.join(documents_dir, "part-*.jsonl")
    jsonl_files = glob.glob(pattern)
    
    if not jsonl_files:
        print(f"디렉토리 '{documents_dir}'에서 part-*.jsonl 파일을 찾을 수 없습니다.")
        return
    
    jsonl_files.sort()  # 파일 순서 정렬
    print(f"발견된 파일들: {jsonl_files}")
    
    all_documents = []
    
    # 모든 JSONL 파일 읽기
    for file_path in jsonl_files:
        print(f"파일 읽는 중: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            file_documents = []
            for line in f:
                file_documents.append(json.loads(line))
            all_documents.extend(file_documents)
            print(f"  - {len(file_documents)}개 문서 로드됨")
    
    print(f"총 {len(all_documents)}개 문서를 색인합니다...")
    
    # 배치로 문서 색인
    bulk_index_documents(es, index_name, all_documents)
    
    print(f"모든 문서 색인 완료!")
    
    # 인덱스 새로고침
    es.indices.refresh(index=index_name)
    
    # 색인된 문서 수 확인
    count = es.count(index=index_name)["count"]
    print(f"색인된 문서 수: {count}")


if __name__ == "__main__":
    # 사용 예시
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    documents_dir = f"{BASE_DIR}/../../data/documents" 
    load_and_index_from_json(documents_dir)