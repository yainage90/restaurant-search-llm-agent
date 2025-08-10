from dotenv import load_dotenv
import os
import json
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import Any


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


def generate_timestamped_index_name() -> str:
    """타임스탬프를 포함한 인덱스 이름 생성"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"coordinates_{timestamp}"


def create_coordinates_index_mapping(es: Elasticsearch, index_name: str) -> None:
    """coordinates 인덱스 및 매핑 생성"""
    mapping = {
        "mappings": {
            "properties": {
                "name": {
                    "type": "text",
                    "analyzer": "index_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword"}
                    }
                },
                "pin": {
                    "properties": {
                        "coordinate": {
                            "type": "geo_point",
                        }
                    }
                },
                "type": {
                    "type": "keyword"
                },
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
   
    es.indices.create(index=index_name, body=mapping)
    print(f"인덱스 '{index_name}' 생성됨")


def preprocess_coordinate_document(doc: dict[str, Any]) -> dict[str, Any]:
    """좌표 문서 전처리"""
    processed_doc = doc.copy()
    
    # geo_point 필드 추가
    processed_doc["pin"] = {
        "coordinate": {
            "lat": processed_doc["lat"],
            "lon": processed_doc["lon"]
        }
    }

    processed_doc.pop("lat")
    processed_doc.pop("lon")
    
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


def get_coordinate_indices(es: Elasticsearch) -> list[str]:
    """coordinates_ 패턴의 모든 색인 목록 반환"""
    try:
        all_indices = es.indices.get_alias(index="coordinates_*", ignore=[404])
        return sorted(list(all_indices.keys())) if all_indices else []
    except Exception:
        return []


def cleanup_old_indices(es: Elasticsearch, current_index: str, backup_count: int = 1) -> None:
    """오래된 색인들을 정리하고 백업 개수만 유지"""
    all_indices = get_coordinate_indices(es)
    
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


def bulk_index_coordinates(es: Elasticsearch, index_name: str, documents: list[dict[str, Any]]) -> None:
    """배치로 좌표 문서들 색인"""
    actions = []
    
    for doc in documents:
        processed_doc = preprocess_coordinate_document(doc)
        
        action = {
            "_index": index_name,
            "_source": processed_doc,
        }
        actions.append(action)
    
    # 배치 크기별로 나누어 색인
    batch_size = 500
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i + batch_size]
        bulk(es, batch)
        print(f"배치 {i//batch_size + 1} 색인 완료 ({len(batch)}개 문서)")


def load_and_index_coordinates(
    alias_name: str = "coordinates",
    backup_count: int = 1,
) -> None:
    """좌표 데이터를 Elasticsearch에 색인 (무중단 배포)"""
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    district_coordinates_file = f"{BASE_DIR}/../../data/coordinates/district_coordinates.jsonl"
    station_coordinates_file = f"{BASE_DIR}/../../data/coordinates/station_coordinates.jsonl"
    
    # Elasticsearch 클라이언트 생성
    es = create_elasticsearch_client()
    
    # 타임스탬프 기반 인덱스 이름 생성
    new_index_name = generate_timestamped_index_name()
    
    print(f"새로운 인덱스 생성: {new_index_name}")
    
    # 새 인덱스 생성
    create_coordinates_index_mapping(es, new_index_name)
    
    all_documents = []
    
    # district_coordinates.jsonl 파일 읽기
    if os.path.exists(district_coordinates_file):
        print(f"파일 읽는 중: {district_coordinates_file}")
        with open(district_coordinates_file, 'r', encoding='utf-8') as f:
            district_documents = []
            for line in f:
                doc = json.loads(line)
                doc['type'] = 'district'
                district_documents.append(doc)
            all_documents.extend(district_documents)
            print(f"  - {len(district_documents)}개 구 좌표 문서 로드됨")
    
    # station_coordinates.jsonl 파일 읽기
    if os.path.exists(station_coordinates_file):
        print(f"파일 읽는 중: {station_coordinates_file}")
        with open(station_coordinates_file, 'r', encoding='utf-8') as f:
            station_documents = []
            for line in f:
                doc = json.loads(line)
                doc['type'] = 'station'
                station_documents.append(doc)
            all_documents.extend(station_documents)
            print(f"  - {len(station_documents)}개 역 좌표 문서 로드됨")
    
    if not all_documents:
        print("색인할 좌표 데이터가 없습니다.")
        return
    
    print(f"총 {len(all_documents)}개 좌표 문서를 색인합니다...")
    
    # 배치로 문서 색인
    bulk_index_coordinates(es, new_index_name, all_documents)
    
    print(f"모든 좌표 문서 색인 완료!")
    
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


def main():
    load_and_index_coordinates()


if __name__ == "__main__":
    main()