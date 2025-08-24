import os
import json
from app.retrieve.nlu import classify_intent_and_extract_entities


natural_queries = [
    {"intent": "search", "query": "강남역 주차되는 일식집"},
    {"intent": "search", "query": "스타벅스 메뉴 알려줘"},
    {"intent": "search", "query": "조용하고 데이트하기 좋은곳"},
    {"intent": "search", "query": "판교 콜키지 되는 술집"},
    {"intent": "search", "query": "정자역 고기집"},
    {"intent": "search", "query": "판교 애견동반 식당 추천"},
    {"intent": "search", "query": "맥도날드 드라이브스루"},
    {"intent": "search", "query": "강남 술집빼고 회식하기 좋은 식당"},
    {"intent": "search", "query": "24시간 감자탕집, 뼈해장국"},
    {"intent": "search", "query": "햄버거. 맥도날드 빼고"},
    {"intent": "information", "query": "마포 진대감 주차되나요?"},
    {"intent": "information", "query": "맥도날드 24시간 하나요?"},
    {"intent": "compare", "query": "강남 진대감 vs 삼육가 어디가 더 맛있어?"},
    {"intent": "compare", "query": "맥도날드 vs 롯데리아 vs 버거킹"},
]

def create_evaluation_queries():
    results = []
    
    for query_data in natural_queries:
        query = query_data["query"]
        
        # NLU 처리
        nlu_result = classify_intent_and_extract_entities(query)
        
        # 평가 데이터 구성
        evaluation_data = {
            "query": query,
            **nlu_result
        }
        
        results.append(evaluation_data)
    
    # JSONL 파일로 저장
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(f"{BASE_DIR}/evaluation_queries.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"총 {len(results)}개의 평가 쿼리를 처리하여 evaluation_queries.jsonl에 저장했습니다.")


if __name__ == "__main__":
    create_evaluation_queries()
