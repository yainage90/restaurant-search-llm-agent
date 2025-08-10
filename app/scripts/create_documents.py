"""
featured_restaurants와 embeddings를 결합하여 최종 documents 생성
"""

import os
import json
from tqdm import tqdm


def load_embeddings(embeddings_file: str) -> dict[str, list[float]]:
    """임베딩 파일을 읽어서 place_id -> embedding 매핑 딕셔너리 생성"""
    embeddings_dict = {}
    with open(embeddings_file, "r", encoding="utf-8") as f:
        for line in f:
            embedding_data = json.loads(line)
            embeddings_dict[embedding_data["place_id"]] = embedding_data["embedding"]
    return embeddings_dict


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FEATURED_DIR = os.path.join(BASE_DIR, "../../data/featured_restaurants")
    EMBEDDINGS_DIR = os.path.join(BASE_DIR, "../../data/embeddings")
    OUTPUT_DIR = os.path.join(BASE_DIR, "../../data/documents")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # featured_restaurants 디렉토리에서 모든 part 파일 찾기
    featured_files = [f for f in os.listdir(FEATURED_DIR) if f.startswith("part-") and f.endswith(".jsonl")]
    featured_files.sort()
    
    for filename in tqdm(featured_files, desc="Processing files"):
        featured_file_path = os.path.join(FEATURED_DIR, filename)
        embeddings_file_path = os.path.join(EMBEDDINGS_DIR, filename)
        output_file_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"Processing {filename}...")
        
        # 임베딩 파일이 없으면 건너뛰기
        if not os.path.exists(embeddings_file_path):
            print(f"Embedding file not found for {filename}, skipping...")
            continue
        
        # 이미 처리된 place_id들 확인
        processed_place_ids = set()
        if os.path.exists(output_file_path):
            with open(output_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    document = json.loads(line)
                    processed_place_ids.add(document["place_id"])
        
        # 임베딩 로드
        print(f"Loading embeddings from {filename}...")
        embeddings_dict = load_embeddings(embeddings_file_path)
        
        # featured_restaurants 파일 읽기 및 임베딩 결합
        documents_to_save = []
        with open(featured_file_path, "r", encoding="utf-8") as f:
            for line in f:
                document = json.loads(line)
                place_id = document["place_id"]
                
                # 이미 처리된 문서는 건너뛰기
                if place_id in processed_place_ids:
                    continue
                
                # 임베딩이 있는 경우에만 문서 생성
                if place_id in embeddings_dict:
                    document["embedding"] = embeddings_dict[place_id]
                    documents_to_save.append(document)
        
        # 결과 저장
        if documents_to_save:
            with open(output_file_path, "a", encoding="utf-8") as f:
                for document in documents_to_save:
                    f.write(f"{json.dumps(document, ensure_ascii=False)}\n")
            print(f"Completed {filename} - created {len(documents_to_save)} documents")
        else:
            print(f"No new documents to create for {filename}")


if __name__ == "__main__":
    main()