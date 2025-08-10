"""
식당 특징 데이터에서 임베딩만 추출하여 저장하는 스크립트
extract_features.py에서 생성된 jsonl 파일들을 읽어서 임베딩만 추출하여 별도 파일로 저장
"""

import os
import json
from tqdm import tqdm
from app.retrieve.embeddings import get_document_embeddings


def process_batch_embeddings(documents: list[dict], batch_size: int = 100) -> list[dict]:
    """
    문서들의 summary를 batch로 임베딩 추출하여 place_id, embedding 형태로 반환
    """
    results = []
    
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
        batch = documents[i:i + batch_size]
        summaries = [doc["summary"] for doc in batch]
        
        # 배치로 임베딩 추출
        embeddings = get_document_embeddings(summaries)
        
        # place_id와 embedding만 저장
        for doc, embedding in zip(batch, embeddings):
            results.append({
                "place_id": doc["place_id"],
                "embedding": embedding
            })
    
    return results


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "../../data/featured_restaurants")
    OUTPUT_DIR = os.path.join(BASE_DIR, "../../data/embeddings")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 입력 디렉토리에서 모든 part 파일 찾기
    input_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("part-") and f.endswith(".jsonl")]
    input_files.sort()  # 파일명 순서로 정렬
    
    for input_filename in tqdm(input_files, desc="Processing files"):
        input_file_path = os.path.join(INPUT_DIR, input_filename)
        output_file_path = os.path.join(OUTPUT_DIR, input_filename)
        
        print(f"Processing {input_filename}...")
        
        # 이미 처리된 place_id들 확인
        processed_place_ids = set()
        if os.path.exists(output_file_path):
            with open(output_file_path, "r", encoding="utf-8") as f_out:
                for line in f_out:
                    embedding_data = json.loads(line)
                    processed_place_ids.add(embedding_data["place_id"])
        
        # 처리할 문서들 로드
        documents_to_process = []
        with open(input_file_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                document = json.loads(line)
                if document["place_id"] not in processed_place_ids:
                    documents_to_process.append(document)
        
        if not documents_to_process:
            print(f"No new documents to process in {input_filename}")
            continue
        
        # 배치로 임베딩 처리
        embedding_results = process_batch_embeddings(documents_to_process, batch_size=100)
        
        # 결과 저장
        with open(output_file_path, "a", encoding="utf-8") as f_out:
            for embedding_data in embedding_results:
                f_out.write(f"{json.dumps(embedding_data, ensure_ascii=False)}\n")
        
        print(f"Completed {input_filename} - processed {len(embedding_results)} embeddings")


if __name__ == "__main__":
    main()