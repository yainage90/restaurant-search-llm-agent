from dotenv import load_dotenv
import os
import json
import glob
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from app.retrieve.embeddings import EMBEDDING_SIZE
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


def read_synonyms(synonyms_file_path: str) -> list[str]:
    """동의어 파일을 읽어서 동의어 리스트 반환"""
    synonyms = []
    if os.path.exists(synonyms_file_path):
        with open(synonyms_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line := line.strip(): # 빈 줄 제외
                    synonyms.append(line)
    return synonyms


def generate_timestamped_index_name() -> str:
    """타임스탬프를 포함한 인덱스 이름 생성"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"restaurants_{timestamp}"


def create_index_mapping(es: Elasticsearch, index_name: str) -> None:
    """Elasticsearch 인덱스 및 매핑 생성"""
    mapping = {
        "mappings": {
            "properties": {
                "place_id": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"}
                    }
                },
                "address": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                },
                "roadAddress": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                },
                "pin": {
                    "properties": {
                        "coordinate": {
                            "type": "geo_point",
                        }
                    }
                },
                "menus": {
                    "type": "nested",
                    "properties": {
                        "name": {
                            "type": "text",
                            "analyzer": "index_analyzer",
                            "search_analyzer": "search_analyzer",
                            "fields": {
                                "raw": {"type": "keyword"},
                            }
                        },
                        "price": {"type": "integer"}
                    }
                },
                "reviews": {"type": "text", "analyzer": "index_analyzer", "search_analyzer": "search_analyzer"},
                "description": {"type": "text", "analyzer": "index_analyzer", "search_analyzer": "search_analyzer"},
                "review_food": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"},
                    }
                },
                "convenience": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"},
                    }
                },
                "atmosphere": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"},
                    }
                },
                "occasion": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"},
                    }
                },
                "features": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"},
                    }
                },
                "summary": {"type": "text", "analyzer": "index_analyzer", "search_analyzer": "search_analyzer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_SIZE,
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
            "analysis": {
                "tokenizer": {
                    "nori": {
                        "type": "nori_tokenizer",
                        "user_dictionary": "analysis/user_dictionary.txt",
                        "decompound_mode": "discard",
                    }
                },
                "filter": {
                    "synonym_filter": {
                        "type": "synonym",
                        "synonyms_path": "analysis/synonyms.txt",
                    }
                },
                "analyzer": {
                    "index_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori",
                        "filter": ["nori_part_of_speech", "lowercase"]
                    },
                    "search_analyzer": {
                        "type": "custom",
                        "tokenizer": "nori",
                        "filter": ["nori_part_of_speech", "lowercase", "synonym_filter"]
                    }
                }
            }
        }
    }
   
    # 새 인덱스 생성 (기존 인덱스 삭제하지 않음)
    es.indices.create(index=index_name, body=mapping)
    print(f"인덱스 '{index_name}' 생성됨")


def preprocess_document(doc: dict[str, Any]) -> dict[str, Any]:
    """문서 전처리"""
    processed_doc = doc.copy()
    
    # 위치 정보 추가 (geo_point 형식)
    processed_doc["pin"] = {
        "coordinate": doc["coordinate"],
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


def update_alias(es: Elasticsearch, alias_name: str, new_index: str) -> None:
    """alias를 새로운 색인으로 업데이트"""
    try:
        # 현재 alias가 가리키는 색인들 확인
        try:
            current_aliases = es.indices.get_alias(name=alias_name)
            current_indices = list(current_aliases.keys())
        except Exception:
            # alias가 존재하지 않는 경우
            current_indices = []
        
        # alias 업데이트
        actions = []
        
        # 기존 alias 제거
        for old_index in current_indices:
            actions.append({"remove": {"index": old_index, "alias": alias_name}})
        
        # 새 alias 추가
        actions.append({"add": {"index": new_index, "alias": alias_name}})
        
        es.indices.update_aliases(body={"actions": actions})
        print(f"alias '{alias_name}'를 '{new_index}'로 업데이트됨")
        
        return current_indices
        
    except Exception as e:
        print(f"alias 업데이트 실패: {e}")
        raise


def get_restaurant_indices(es: Elasticsearch) -> list[str]:
    """restaurants_ 패턴의 모든 색인 목록 반환"""
    try:
        all_indices = es.indices.get_alias(index="restaurants_*", ignore=[404])
        return sorted(list(all_indices.keys())) if all_indices else []
    except Exception:
        return []


def cleanup_old_indices(es: Elasticsearch, current_index: str, backup_count: int = 1) -> None:
    """오래된 색인들을 정리하고 백업 개수만 유지"""
    all_indices = get_restaurant_indices(es)
    
    # 현재 색인 제외
    old_indices = [idx for idx in all_indices if idx != current_index]
    
    # 타임스탬프 기준 정렬 (최신순)
    old_indices.sort(reverse=True)
    
    # 백업으로 유지할 색인들과 삭제할 색인들 구분
    keep_indices = old_indices[:backup_count]
    delete_indices = old_indices[backup_count:]
    
    print(f"백업으로 유지할 색인: {keep_indices}")
    
    # 오래된 색인들 삭제
    for index_name in delete_indices:
        try:
            es.indices.delete(index=index_name)
            print(f"오래된 색인 '{index_name}' 삭제됨")
        except Exception as e:
            print(f"색인 '{index_name}' 삭제 실패: {e}")


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
    batch_size = 500
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i + batch_size]
        bulk(es, batch)
        print(f"배치 {i//batch_size + 1} 색인 완료 ({len(batch)}개 문서)")


def load_and_index_from_json(
    documents_dir: str, 
    alias_name: str = "restaurants",
    backup_count: int = 1,
) -> None:
    """documents 디렉토리의 part-*.jsonl 파일들에서 완성된 문서 데이터를 읽어 Elasticsearch에 색인 (무중단 배포)"""
    
    # Elasticsearch 클라이언트 생성
    es = create_elasticsearch_client()
    
    # 타임스탬프 기반 인덱스 이름 생성
    new_index_name = generate_timestamped_index_name()
    
    print(f"새로운 인덱스 생성: {new_index_name}")
    
    # 새 인덱스 생성
    create_index_mapping(es, new_index_name)
    
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
    bulk_index_documents(es, new_index_name, all_documents)
    
    print(f"모든 문서 색인 완료!")
    
    # 인덱스 새로고침
    es.indices.refresh(index=new_index_name)
    
    # 색인된 문서 수 확인
    count = es.count(index=new_index_name)["count"]
    print(f"색인된 문서 수: {count}")
    
    # alias 업데이트 (무중단 배포)
    print(f"alias '{alias_name}' 업데이트 중...")
    update_alias(es, alias_name, new_index_name)
    
    # 오래된 인덱스 정리
    print(f"오래된 인덱스 정리 중... (백업 {backup_count}개 유지)")
    cleanup_old_indices(es, new_index_name, backup_count)


if __name__ == "__main__":
    # 사용 예시
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    documents_dir = f"{BASE_DIR}/../../data/documents" 
    load_and_index_from_json(documents_dir)