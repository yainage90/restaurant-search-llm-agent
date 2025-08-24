import json
from pathlib import Path
from typing import Any
from pydantic import BaseModel
from app.retrieve.hybrid_search import execute_bm25_search, execute_vector_search, reciprocal_rank_fusion
from app.llm.llm import generate_with_gemini, generate_with_openai


class SearchEvaluation(BaseModel):
    search_method: str
    quality: str  # excellent, good, fair, poor
    relevance: str  # highly_relevant, relevant, partially_relevant, not_relevant
    reasoning: str
    score: int  # 1-10


class QueryEvaluationResult(BaseModel):
    query: str
    bm25_evaluation: SearchEvaluation
    vector_evaluation: SearchEvaluation
    rrf_evaluation: SearchEvaluation


def load_evaluation_queries() -> list[dict[str, Any]]:
    """evaluation_queries.jsonl 파일 불러오기"""
    queries = []
    queries_path = Path(__file__).parent / "evaluation_queries.jsonl"
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    
    return queries


def perform_searches(query_data: dict[str, Any], k: int = 5) -> tuple[list[dict], list[dict], list[dict]]:
    """각 검색 방법으로 검색 수행 - suggested_queries를 사용하여 실제 서비스와 동일하게"""
    query = query_data["query"]
    entities = query_data["entities"]
    negation_entities = query_data["negation_entities"]
    intent = query_data["intent"]
    suggested_queries = query_data.get("suggested_queries", [query])
    
    print(f"  원본 쿼리: {query}")
    print(f"  재정의된 쿼리: {suggested_queries}")
    
    # suggested_queries 전체를 사용하여 BM25, 벡터 검색 수행 (쿼리별 균등 분배)
    search_size = k * 10
    
    # 쿼리별로 개별 검색 수행
    query_results = []
    for sq in suggested_queries:
        if len(suggested_queries) > 1:
            guaranteed_results = max(1, search_size // len(suggested_queries))
            individual_search_size = max(20, guaranteed_results * 4)
        else:
            individual_search_size = search_size
            
        bm25_res = execute_bm25_search(sq, entities, negation_entities, intent, individual_search_size)
        vector_res = execute_vector_search(sq, entities, negation_entities, intent, individual_search_size)
        
        # 디버그: 각 쿼리별 결과 확인
        print(f"    '{sq}' 검색 결과: BM25={len(bm25_res)}개, 벡터={len(vector_res)}개")
        if bm25_res:
            print(f"      BM25 상위 3개: {[doc.get('title', 'N/A') for doc in bm25_res[:3]]}")
        if vector_res:
            print(f"      벡터 상위 3개: {[doc.get('title', 'N/A') for doc in vector_res[:3]]}")
        
        # 쿼리별 RRF 결합
        query_rrf_results = reciprocal_rank_fusion(bm25_res, vector_res)
        
        query_results.append({
            'query': sq,
            'bm25_results': bm25_res,
            'vector_results': vector_res,
            'rrf_results': query_rrf_results
        })
    
    # RRF도 라운드로빈 방식으로 균등 분배
    final_results = []
    seen_ids = set()

    if len(query_results) == 1:
        final_results = query_results[0]['rrf_results']
    else:
        results_per_query = max(1, search_size // len(query_results))

        # 쿼리별로 results_per_query개씩 선택
        rrf_by_query = []
        for query_data in query_results:
            query_name = query_data['query']
            query_rrf = []
            added_count = 0
            for result in query_data['rrf_results']:
                if added_count >= results_per_query:
                    break
                if result['place_id'] not in seen_ids:
                    query_rrf.append(result)
                    seen_ids.add(result['place_id'])
                    added_count += 1
            rrf_by_query.append(query_rrf)
        
        # 라운드로빈 방식으로 결합
        max_results = max(len(qr) for qr in rrf_by_query) if rrf_by_query else 0
        for i in range(max_results):
            for query_results_list in rrf_by_query:
                if i < len(query_results_list):
                    final_results.append(query_results_list[i])
        
        # 부족한 경우 추가 선택
        if len(final_results) < search_size:
            for query_data in query_results:
                if len(final_results) >= search_size:
                    break
                for result in query_data['rrf_results']:
                    if len(final_results) >= search_size:
                        break
                    if result['place_id'] not in seen_ids:
                        final_results.append(result)
                        seen_ids.add(result['place_id'])
    
    # 평가용으로 각 검색 방법별 결과도 균등 분배로 준비
    final_bm25_results = []
    final_vector_results = []
    seen_bm25_ids = set()
    seen_vector_ids = set()
    
    if len(query_results) == 1:
        final_bm25_results = query_results[0]['bm25_results']
        final_vector_results = query_results[0]['vector_results']
    else:
        results_per_query = max(1, search_size // len(query_results))
        
        # BM25, 벡터를 라운드로빈 방식으로 균등 분배
        bm25_by_query = []
        vector_by_query = []
        
        for query_data in query_results:
            query_name = query_data['query']
            
            # 각 쿼리별로 results_per_query개씩 선택
            query_bm25 = []
            added_bm25 = 0
            for doc in query_data['bm25_results']:
                if added_bm25 >= results_per_query:
                    break
                if doc['place_id'] not in seen_bm25_ids:
                    query_bm25.append(doc)
                    seen_bm25_ids.add(doc['place_id'])
                    added_bm25 += 1
            bm25_by_query.append(query_bm25)
                    
            query_vector = []
            added_vector = 0
            for doc in query_data['vector_results']:
                if added_vector >= results_per_query:
                    break
                if doc['place_id'] not in seen_vector_ids:
                    query_vector.append(doc)
                    seen_vector_ids.add(doc['place_id'])
                    added_vector += 1
            vector_by_query.append(query_vector)
        
        # 라운드로빈 방식으로 결합
        max_results = max(len(qr) for qr in bm25_by_query) if bm25_by_query else 0
        for i in range(max_results):
            for query_results_list in bm25_by_query:
                if i < len(query_results_list):
                    final_bm25_results.append(query_results_list[i])
        
        max_results = max(len(qr) for qr in vector_by_query) if vector_by_query else 0
        for i in range(max_results):
            for query_results_list in vector_by_query:
                if i < len(query_results_list):
                    final_vector_results.append(query_results_list[i])
        
        # 부족한 경우 추가 선택
        if len(final_bm25_results) < search_size:
            for query_data in query_results:
                if len(final_bm25_results) >= search_size:
                    break
                for doc in query_data['bm25_results']:
                    if len(final_bm25_results) >= search_size:
                        break
                    if doc['place_id'] not in seen_bm25_ids:
                        final_bm25_results.append(doc)
                        seen_bm25_ids.add(doc['place_id'])
        
        if len(final_vector_results) < search_size:
            for query_data in query_results:
                if len(final_vector_results) >= search_size:
                    break
                for doc in query_data['vector_results']:
                    if len(final_vector_results) >= search_size:
                        break
                    if doc['place_id'] not in seen_vector_ids:
                        final_vector_results.append(doc)
                        seen_vector_ids.add(doc['place_id'])
    
    bm25_results = final_bm25_results[:search_size]
    vector_results = final_vector_results[:search_size]
    rrf_results = final_results[:search_size]
    
    # 디버그: 최종 반환 결과 확인 (상위 k개)
    print(f"  최종 반환 결과 (상위 {k}개):")
    if bm25_results[:k]:
        print(f"    BM25 상위 {k}개: {[doc.get('title', 'N/A') for doc in bm25_results[:k]]}")
    if vector_results[:k]:
        print(f"    벡터 상위 {k}개: {[doc.get('title', 'N/A') for doc in vector_results[:k]]}")
    if rrf_results[:k]:
        print(f"    RRF 상위 {k}개: {[doc.get('title', 'N/A') for doc in rrf_results[:k]]}")
    
    return bm25_results[:k], vector_results[:k], rrf_results[:k]


def format_results_for_evaluation(results: list[dict], method_name: str) -> str:
    """결과를 평가용 텍스트로 포맷"""
    if not results:
        return f"{method_name} 결과: 검색 결과가 없습니다."
    
    formatted = f"{method_name} 결과 (상위 {len(results)}개):\n\n"
    
    for doc in results:
        formatted += f"```\n{doc.get('summary', 'N/A')}\n"
    
    return formatted


def evaluate_search_results(query: str, results: list[dict], method_name: str) -> SearchEvaluation:
    """LLM을 사용해 검색 결과 평가"""
    
    system_prompt = """
당신은 음식점 검색 시스템의 품질을 평가하는 전문가입니다.
주어진 쿼리와 검색 결과는 LLM 입력으로 사용하여 답변을 생성하는데 사용합니다.
주어진 사용자 쿼리와 검색 결과에 따라 다음 기준으로 점수를 매겨주세요:

평가 기준:
1. quality (품질):
   - excellent: 매우 정확하고 유용한 결과
   - good: 좋은 결과, 대부분 유용함
   - fair: 보통 수준, 일부 유용함
   - poor: 부정확하거나 유용하지 않음

2. relevance (관련성):
   - highly_relevant: 쿼리와 매우 관련성이 높음
   - relevant: 쿼리와 관련성이 있음
   - partially_relevant: 부분적으로 관련성이 있음
   - not_relevant: 쿼리와 관련성이 없음

3. score: 1-10점 (10점이 최고)

4. reasoning: 평가 근거를 상세히 설명

평가 근거(reasoning)은 한글로 간결하게 핵심만 작성해주세요.
"""
    
    user_prompt = f"""
사용자 쿼리: "{query}"
검색 방법: {method_name}

{format_results_for_evaluation(results, method_name)}

위 검색 결과를 평가해주세요.
"""
    
    try:
        result = generate_with_openai(
        # result = generate_with_gemini(
            model="gpt-5-mini",
            # model="gemini-2.5-flash",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=1024,
            text_format=SearchEvaluation,
            # response_shcema=SearchEvaluation,
        )
        
        return SearchEvaluation(**result)
    
    except Exception as e:
        print(f"평가 중 오류 발생: {e}")
        return SearchEvaluation(
            search_method=method_name,
            quality="fair",
            relevance="partially_relevant",
            reasoning=f"평가 중 오류 발생: {str(e)}",
            score=5
        )


def evaluate_single_query(query_data: dict[str, Any], k: int = 5) -> tuple[QueryEvaluationResult, list[dict], list[dict], list[dict]]:
    """단일 쿼리에 대한 평가 수행"""
    query = query_data["query"]
    print(f"\n평가 중: '{query}'")
    
    # 검색 수행
    bm25_results, vector_results, rrf_results = perform_searches(query_data, k=k)
    
    print(f"  BM25: {len(bm25_results)}개 결과")
    print(f"  벡터: {len(vector_results)}개 결과")
    print(f"  RRF: {len(rrf_results)}개 결과")
    
    # 각 방법별 평가
    bm25_eval = evaluate_search_results(query, bm25_results, "BM25")
    print(f"  BM25 평가 완료: {bm25_eval.quality}/{bm25_eval.relevance} ({bm25_eval.score}/10)")
    
    vector_eval = evaluate_search_results(query, vector_results, "벡터검색")
    print(f"  벡터 평가 완료: {vector_eval.quality}/{vector_eval.relevance} ({vector_eval.score}/10)")
    
    rrf_eval = evaluate_search_results(query, rrf_results, "RRF")
    print(f"  RRF 평가 완료: {rrf_eval.quality}/{rrf_eval.relevance} ({rrf_eval.score}/10)")
    
    result = QueryEvaluationResult(
        query=query,
        bm25_evaluation=bm25_eval,
        vector_evaluation=vector_eval,
        rrf_evaluation=rrf_eval
    )
    
    return result, bm25_results, vector_results, rrf_results


def initialize_evaluation_files(total_queries: int) -> tuple[Path, Path]:
    """평가 결과 파일들 초기화"""
    result_path = Path(__file__).parent / "evaluation_result.md"
    docs_path = Path(__file__).parent / "retrieved_docs.md"
    
    # evaluation_result.md - 평가 점수와 요약만
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write("# 임베딩 검색 품질 평가 결과\n\n")
        f.write(f"총 {total_queries}개 쿼리에 대한 평가 결과입니다.\n\n")
        f.write("## 개별 쿼리 평가 점수\n\n")
    
    # retrieved_docs.md - 검색 결과 문서들
    with open(docs_path, 'w', encoding='utf-8') as f:
        f.write("# 검색 결과 문서\n\n")
        f.write(f"총 {total_queries}개 쿼리에 대한 검색 결과입니다.\n\n")
    
    return result_path, docs_path


def write_evaluation_result(result: QueryEvaluationResult, query_index: int, result_path: Path):
    """평가 점수를 evaluation_result.md에 추가"""
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(f"### {query_index}. {result.query}\n\n")
        
        f.write("| 검색방법 | 품질 | 관련성 | 점수 | 평가근거 |\n")
        f.write("|----------|------|--------|------|----------|\n")
        f.write(f"| BM25 | {result.bm25_evaluation.quality} | {result.bm25_evaluation.relevance} | {result.bm25_evaluation.score}/10 | {result.bm25_evaluation.reasoning} |\n")
        f.write(f"| 벡터검색 | {result.vector_evaluation.quality} | {result.vector_evaluation.relevance} | {result.vector_evaluation.score}/10 | {result.vector_evaluation.reasoning} |\n")
        f.write(f"| RRF | {result.rrf_evaluation.quality} | {result.rrf_evaluation.relevance} | {result.rrf_evaluation.score}/10 | {result.rrf_evaluation.reasoning} |\n\n")


def write_retrieved_docs(query_index: int, query: str, bm25_results: list[dict], vector_results: list[dict], rrf_results: list[dict], docs_path: Path):
    """검색 결과 문서들을 retrieved_docs.md에 추가"""
    with open(docs_path, 'a', encoding='utf-8') as f:
        f.write(f"## {query_index}. {query}\n\n")
        
        # BM25 결과
        f.write("### BM25 검색 결과\n\n")
        f.write(format_results_for_evaluation(bm25_results, "BM25"))
        f.write("\n\n")
        
        # 벡터 검색 결과
        f.write("### 벡터 검색 결과\n\n")
        f.write(format_results_for_evaluation(vector_results, "벡터검색"))
        f.write("\n\n")
        
        # RRF 결과
        f.write("### RRF (Reciprocal Rank Fusion) 결과\n\n")
        f.write(format_results_for_evaluation(rrf_results, "RRF"))
        f.write("\n\n---\n\n")


def write_final_summary(all_results: list[QueryEvaluationResult], queries: list[dict[str, Any]], result_path: Path):
    """모든 평가 완료 후 요약 정보 추가"""
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write("## 전체 요약\n\n")
        
        bm25_scores = [r.bm25_evaluation.score for r in all_results]
        vector_scores = [r.vector_evaluation.score for r in all_results]
        rrf_scores = [r.rrf_evaluation.score for r in all_results]
        
        f.write(f"- **BM25 평균 점수**: {sum(bm25_scores)/len(bm25_scores):.2f}/10\n")
        f.write(f"- **벡터검색 평균 점수**: {sum(vector_scores)/len(vector_scores):.2f}/10\n")
        f.write(f"- **RRF 평균 점수**: {sum(rrf_scores)/len(rrf_scores):.2f}/10\n\n")
        
        # Intent별 평균 점수 계산
        f.write("## Intent별 평균 점수\n\n")
        
        intent_scores = {}
        for i, result in enumerate(all_results):
            intent = queries[i]["intent"]
            if intent not in intent_scores:
                intent_scores[intent] = {"bm25": [], "vector": [], "rrf": []}
            
            intent_scores[intent]["bm25"].append(result.bm25_evaluation.score)
            intent_scores[intent]["vector"].append(result.vector_evaluation.score)
            intent_scores[intent]["rrf"].append(result.rrf_evaluation.score)
        
        f.write("| Intent | BM25 | 벡터검색 | RRF |\n")
        f.write("|--------|------|----------|-----|\n")
        
        for intent, scores in intent_scores.items():
            bm25_avg = sum(scores["bm25"]) / len(scores["bm25"])
            vector_avg = sum(scores["vector"]) / len(scores["vector"])
            rrf_avg = sum(scores["rrf"]) / len(scores["rrf"])
            
            f.write(f"| {intent} | {bm25_avg:.2f}/10 | {vector_avg:.2f}/10 | {rrf_avg:.2f}/10 |\n")
        
        f.write("\n")


def main():
    """메인 평가 실행 함수"""
    print("임베딩 검색 품질 평가를 시작합니다...")
    
    # 1. evaluation_queries.jsonl 파일 불러오기
    queries = load_evaluation_queries()
    print(f"총 {len(queries)}개의 평가 쿼리를 불러왔습니다.")
    
    # 2. 결과 파일들 초기화
    result_path, docs_path = initialize_evaluation_files(len(queries))
    print(f"결과 파일들 초기화: {result_path}, {docs_path}")
    
    # 3. 각 쿼리에 대해 평가 수행하고 즉시 파일에 작성
    evaluation_results = []
    
    for i, query_data in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] 평가 진행 중...")
        result, bm25_results, vector_results, rrf_results = evaluate_single_query(query_data, k=5)
        evaluation_results.append(result)
        
        # 즉시 파일들에 결과 작성
        write_evaluation_result(result, i, result_path)
        write_retrieved_docs(i, query_data["query"], bm25_results, vector_results, rrf_results, docs_path)
        print(f"  결과를 파일들에 작성했습니다.")
    
    # 4. 전체 요약 정보 추가
    write_final_summary(evaluation_results, queries, result_path)
    
    print(f"\n평가가 완료되었습니다!")
    print(f"평가 결과: {result_path}")
    print(f"검색 문서: {docs_path}")


if __name__ == "__main__":
    main()