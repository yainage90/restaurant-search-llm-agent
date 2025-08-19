"""
검색 의도 분류 및 엔티티 추출 모듈
"""

from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal, Any

load_dotenv()

client = genai.Client()

SYSTEM_PROMPT = """
당신은 식당 검색을 위한 의도 분류 및 엔티티 추출 전문가입니다.
사용자의 자연어 쿼리를 분석하여 검색 의도를 분류하고 필요한 엔티티를 추출하는 것이 목표입니다.

규칙:
1. 반드시 JSON 형태로만 응답하세요.
2. 정확한 정보만 추출하세요.
3. 값이 없을 경우 빈 배열 [] 혹은 빈 문자열("")을 출력해주세요.
4. 반드시 쿼리에 등장한 키워드만 추출하세요.
"""

USER_QUERY_PROMPT = """
다음 자연어 쿼리를 분석하여 의도를 분류하고 엔티티를 추출한 후 검색에 최적화된 쿼리를 재정의해주세요:

의도 분류:
- search: 일반적인 검색 (예: "강남역 일식집 추천", "주차되는 식당")
- compare: 여러 식당 비교 (예: "A와 B 중 어디가 더 맛있어?", "버거킹과 맥도날드 비교")
- information: 특정 식당 정보 요청 (예: "진대감 영업시간", "버거킹 메뉴")

엔티티 추출:
- location: 위치 정보 (지역명, 식당명. 예: 정자역, 마포)
- title: 식당명 (비교나 정보 요청 시 중요)
- menu: 메뉴명(예: 국밥, 치킨, 회, 돈가스, 파스타)
- category: 식당 카테고리(예: 한식, 일식, 중식, 양식, 퓨전요리)
- convenience: 편의사항(주차|발렛|배달|포장|예약|룸|콜키지|반려동물|와이파이|24시|구워줌)
- atmosphere: 분위기(예: 이국적인, 색다른, 로맨틱한)
- occasion: 상황(예: 회식, 단체, 데이트, 혼밥, 가족)

쿼리 재정의 규칙:
- search: 1개의 포괄적 검색 쿼리
- compare: 각 비교 대상별 개별 쿼리 생성
- information: 1개의 정보 요청에 최적화된 쿼리

예시:
1. "강남역 주차되는 일식집" 
   -> intent: "search", entities: {{"location": ["강남역"], "category": ["일식"], "convenience": ["주차"]}}, suggested_queries: ["강남역 일식 주차"]

2. "버거킹과 맥도날드 중 어디가 더 맛있어?"
   -> intent: "compare", entities: {{"title": ["버거킹", "맥도날드"]}}, suggested_queries: ["버거킹", "맥도날드"]

3. "마포 진대감 주차되나요?"
   -> intent: "information", entities: {{"location": ["마포"], "title": ["진대감"], "convenience": ["주차"]}}, suggested_queries: ["마포 진대감 주차"]

자연어 쿼리: {query}
"""


class IntentResult(BaseModel):
    intent: Literal["search", "compare", "information"]
    entities: Any
    suggested_queries: list[str]


def classify_intent_and_extract_entities(query: str) -> dict:
    """쿼리의 의도를 분류하고 엔티티를 추출"""
    user_prompt = USER_QUERY_PROMPT.format(query=query)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=IntentResult,
            max_output_tokens=512,
        )
    )
    
    result = response.parsed.model_dump(exclude_none=True)
    # 빈 값들 제거
    result["entities"] = {k: v for k, v in result["entities"].items() if v}
    
    return result


def test_intent_classification():
    """의도 분류 및 엔티티 추출 테스트"""
    test_cases = [
        "강남역 주차되는 일식집",
        "마포 진대감 주차되나요?",
        "강남 진대감 vs 삼육가",
        "판교 애견동반 식당 추천",
        "스타벅스 메뉴 알려줘",
        "조용한 데이트 장소 추천",
        "판교 콜키지 되는 술집",
        "정자역 고깃집",
    ]
    
    print("=== 의도 분류 및 엔티티 추출 테스트 ===")
    for i, query in enumerate(test_cases, 1):
        result = classify_intent_and_extract_entities(query)
        print(f"\n{i}. 쿼리: '{query}'")
        print(f"   의도: {result['intent']}")
        print(f"   엔티티: {result['entities']}")


if __name__ == "__main__":
    test_intent_classification()