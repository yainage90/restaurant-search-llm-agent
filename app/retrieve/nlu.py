"""
검색 의도 분류 및 엔티티 추출 모듈
"""

from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal, Any
from app.llm.llm import generate_with_gemini

load_dotenv()


SYSTEM_PROMPT = """
당신은 식당 검색을 위한 의도 분류 및 엔티티 추출 전문가입니다.
사용자의 자연어 쿼리를 분석하여 검색 의도를 분류하고 필요한 엔티티를 추출하는 것이 목표입니다.

규칙:
1. 반드시 JSON 형태로만 응답하세요.
2. 정확한 정보만 추출하세요.
3. 값이 없을 경우 빈 배열 [] 혹은 빈 문자열("")을 출력해주세요.
4. 반드시 쿼리에 등장한 키워드만 추출하세요.
5. "빼고", "제외하고", "아닌" 등 부정의 의미가 담긴 키워드는 negation_entities로 따로 추출하세요.
"""

USER_QUERY_PROMPT = """
다음 자연어 쿼리를 분석하여 의도를 분류하고 엔티티를 추출한 후 검색에 최적화된 쿼리를 재정의해주세요:

중요: 만약 "이전 요청:"이 포함된 경우, 이전 요청의 맥락을 고려하여 현재 요청을 이해하고 통합된 결과를 제공하세요.

의도 분류:
- search: 일반적인 검색 (예: "강남역 일식집 추천", "주차되는 식당")
- compare: 여러 식당 비교 (예: "A와 B 중 어디가 더 맛있어?", "버거킹과 맥도날드 비교")
- information: 특정 식당 정보 요청 (예: "진대감 영업시간", "버거킹 메뉴")

엔티티 추출 (entities, negation_entities):
- location: 위치 정보 (지역명, 식당명. 예: 정자역, 마포)
- title: 식당명 (비교나 정보 요청 시 중요)
- menu: 메뉴명(예: 국밥, 치킨, 회, 돈가스, 파스타)
- category: 식당 카테고리(예: 한식, 일식, 중식, 양식, 퓨전요리, 술집)
- convenience: 편의사항(주차|발렛|배달|포장|예약|룸|콜키지|반려동물|와이파이|24시|구워줌)
- atmosphere: 분위기(예: 이국적인, 색다른, 로맨틱한)
- occasion: 상황(예: 회식, 단체, 데이트, 혼밥, 가족)

쿼리 재정의 규칙:
- search: 1개의 포괄적 검색 쿼리
- compare: 각 비교 대상별 개별 쿼리 생성
- information: 1개의 정보 요청에 최적화된 쿼리

예시:
1. "강남역 주차되는 일식집" 
   -> intent: "search", entities: {{"location": ["강남역"], "category": ["일식"], "convenience": ["주차"]}}, negation_entities: {{}}, suggested_queries: ["강남역 일식 주차"]

2. "버거킹과 맥도날드 중 어디가 더 맛있어?"
   -> intent: "compare", entities: {{"title": ["버거킹", "맥도날드"]}}, negation_entities: {{}}, suggested_queries: ["버거킹", "맥도날드"]

3. "마포 진대감 주차되나요?"
   -> intent: "information", entities: {{"location": ["마포"], "title": ["진대감"], "convenience": ["주차"]}}, negation_entities: {{}}, suggested_queries: ["마포 진대감 주차"]

4. "강남역 주변 회식하기 좋은 한식당. 술집 빼고"
   -> intent: "search", entities: {{"location": ["강남역"], "occasion": ["회식"], "category": ["한식"]}}, negation_entities: {{"category": ["술집"]}}, suggested_queries: ["강남역 회식 한식당"]

5. 맥락 처리 예시:
   "이전 요청: 강남역 주변 회식하기 좋은 식당 추천해줘
    현재 요청: 중국집 빼고"
   -> intent: "search", entities: {{"location": ["강남역"], "occasion": ["회식"]}}, negation_entities: {{"category": ["중국집"]}}, suggested_queries: ["강남역 회식 식당"]

자연어 쿼리: {query}
"""


class IntentResult(BaseModel):
    intent: Literal["search", "compare", "information"]
    entities: Any
    negation_entities: Any
    suggested_queries: list[str]


def classify_intent_and_extract_entities(query: str, context: str = None) -> dict:
    """쿼리의 의도를 분류하고 엔티티를 추출"""
    # 맥락이 있으면 이전 쿼리와 현재 쿼리를 결합
    if context:
        combined_query = f"이전 요청: {context}\n현재 요청: {query}"
        user_prompt = USER_QUERY_PROMPT.format(query=combined_query)
    else:
        user_prompt = USER_QUERY_PROMPT.format(query=query)
    
    result = generate_with_gemini(
        model="gemini-2.5-flash-lite",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_output_tokens=512,
        response_shcema=IntentResult,
    )
    
    # 빈 값들 제거
    result["entities"] = {k: v for k, v in result.get("entities", {}).items() if v}
    
    if "negation_entities" in result and result["negation_entities"]:
        result["negation_entities"] = {k: v for k, v in result["negation_entities"].items() if v}
    else:
        result["negation_entities"] = {}
        
    return result


def test_intent_classification():
    """의도 분류 및 엔티티 추출 테스트"""
    test_cases = [
        ("강남역 주차되는 일식집", None),
        ("마포 진대감 주차되나요?", None),
        ("강남 진대감 vs 삼육가", None),
        ("판교 애견동반 식당 추천", None),
        ("스타벅스 메뉴 알려줘", None),
        ("조용한 데이트 장소 추천", None),
        ("판교 콜키지 되는 술집", None),
        ("정자역 고깃집", None),
    ]
    
    # 맥락 처리 테스트 케이스 추가
    context_test_cases = [
        ("중국집 빼고", "강남역 주변 회식하기 좋은 식당 추천해줘"),
        ("주차되는 곳으로", "이태원 맛집 추천해줘"),
        ("일식은 어때?", "판교 데이트 장소 추천"),
    ]
    
    print("=== 의도 분류 및 엔티티 추출 테스트 ===")
    for i, (query, context) in enumerate(test_cases, 1):
        result = classify_intent_and_extract_entities(query, context)
        print(f"\n{i}. 쿼리: '{query}'")
        print(f"   의도: {result['intent']}")
        print(f"   엔티티: {result['entities']}")
    
    print("\n=== 맥락 처리 테스트 ===")
    for i, (query, context) in enumerate(context_test_cases, 1):
        result = classify_intent_and_extract_entities(query, context)
        print(f"\n{i}. 이전 요청: '{context}'")
        print(f"   현재 요청: '{query}'")
        print(f"   의도: {result['intent']}")
        print(f"   엔티티: {result['entities']}")
        print(f"   부정 엔티티: {result['negation_entities']}")
        print(f"   제안 쿼리: {result['suggested_queries']}")


if __name__ == "__main__":
    test_intent_classification()
