"""
쿼리 재작성 코드 작성.
1. 프롬프트 작성
2. 하나하나 호출할 수 있는 코드 작성
3. 테스트 함수 작성 후 __name__ = "__main__"으로 실행하면 테스트 함수 실행해서 결과 출력
"""

import json
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = genai.Client()

SYSTEM_PROMPT = """
당신은 식당 검색을 위한 쿼리 재작성 전문가입니다.
사용자의 자연어 쿼리를 구조화된 JSON 형태로 변환하는 것이 목표입니다.

규칙:
1. 반드시 JSON 형태로만 응답하세요
2. 언급되지 않은 필드는 생략하세요
3. 정확한 정보만 추출하고 추측하지 마세요
4. location의 relation은 정확히 "exact" 또는 "nearby"만 사용하세요

응답 형식:
{
  "location": [{"name": "지역명", "relation": "exact|nearby"}],
  "cuisine": ["요리 종류"],
  "menu": ["메뉴명"],
  "convenience": ["편의 사항"],
  "atmosphere": ["분위기"],
  "occasion": ["상황"]
}
"""

USER_QUERY_PROMPT = """
다음 자연어 쿼리를 구조화된 JSON으로 변환해주세요:

구조화된 쿼리의 필드:
- location: 위치 정보
  - name: 지역명(강남역, 정자동, 타임스퀘어)
  - relation: 관련성 (exact: 특정 식당명, nearby: 근처)
- cuisine: 요리(예: 한식, 일식, 중식, 양식, 퓨전요리)
- menu: 메뉴명(예: 국밥, 치킨, 회, 돈가스, 파스타)
- convenience: 편의(예: 주차, 배달, 포장, 예약, 룸, 반려동물, 고기구워주는)
- atmosphere: 분위기(예: 이국적인, 색다른, 로맨틱한, 조용한)
- occasion: 상황(예: 회식, 단체, 데이트, 혼밥, 가족)

예시:
1. "강남역 주차되는 일식집" -> {{"location": [{{"name": "강남역", "relation": "nearby"}}], "cuisine": ["일식"], "convenience": ["주차"]}}
2. "판교 애견동반 식당" -> {{"location": [{{"name": "판교", "relation": "nearby"}}], "convenience": ["반려동물"]}}
3. "마포 진대감 주차되나요?" -> {{"location": [{{"name": "마포", "relation": "nearby"}}, {{"name": "진대감", "relation": "exact"}}], "convenience": ["주차"]}}

실제 자연어 쿼리에 포함된 키워드에 관련한 것만 출력해주세요. 예측해서 추가하면 안됩니다.

자연어 쿼리: {query}
"""


class Location(BaseModel):
    name: str
    relation: str


class StructuredQuery(BaseModel):
    location: list[Location] | None
    cuisine: list[str] | None
    menu: list[str] | None
    convenience: list[str] | None
    atmosphere: list[str] | None
    occasion: list | None


def rewrite_query(query: str) -> dict:
    """LLM을 사용한 쿼리 재작성"""
    user_prompt = USER_QUERY_PROMPT.format(query=query)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=StructuredQuery,
            max_output_tokens=128,
        )
    )
    structured_query = response.parsed.model_dump(exclude_none=True)
    structured_query = {k: v for k, v in structured_query.items() if v}
    
    return structured_query


def test_query_rewrite():
    """README.md의 예시를 사용한 테스트 함수"""
    test_cases = [
        {
            "query": "강남역 주차되는 일식집",
            "expected": {
                "location": [{"name": "강남역", "relation": "nearby"}],
                "cuisine": ["일식"],
                "convenience": ["주차"]
            }
        },
        {
            "query": "판교 애견동반 식당",
            "expected": {
                "location": [{"name": "판교", "relation": "nearby"}],
                "convenience": ["반려동물"]
            }
        },
        {
            "query": "마포 진대감 주차되나요?",
            "expected": {
                "location": [{"name": "마포", "relation": "nearby"}, {"name": "진대감", "relation": "exact"}],
                "convenience": ["주차"]
            }
        },
        {
            "query": "조용히 대화할 수 있는 맥주집",
            "expected": {
                "menu": ["맥주"],
                "atmosphere": ["조용한"]
            }
        },
        {
            "query": "홍대에 회식하기 좋은 삼겹살집 추천해줘",
            "expected": {
                "location": [{"name": "홍대", "relation": "nearby"}],
                "menu": ["삼겹살"],
                "occasion": ["회식"]
            }
        }
    ]
    
    print("=== 쿼리 재작성 테스트 ===")
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        expected = test_case["expected"]
        result = rewrite_query(query)
        
        print(f"\n{i}. 테스트 쿼리: '{query}'")
        print(f"   예상 결과: {expected}")
        print(f"   실제 결과: {result}")
        print(f"   매칭 여부: {'✓' if result == expected else '✗'}")

if __name__ == "__main__":
    test_query_rewrite()
