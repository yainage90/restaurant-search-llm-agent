from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

client = genai.Client()

SYSTEM_PROMPT = """
당신은 식당 검색을 위한 쿼리 재작성 전문가입니다.
사용자의 자연어 쿼리를 구조화된 JSON 형태로 변환하는 것이 목표입니다.

규칙:
1. 반드시 JSON 형태로만 응답하세요.
2. 정확한 정보만 추출하세요.
3. location의 relation은 정확히 "exact" 또는 "nearby"만 사용하세요
4. 값이 없을 경우 빈 배열 [] 혹은 빈 문자열("")을 출력해주세요.
5. 반드시 쿼리에 등장한 키워드만 추출하세요.
"""

USER_QUERY_PROMPT = """
다음 자연어 쿼리를 구조화된 JSON으로 변환해주세요:

구조화된 쿼리의 필드:
- location: 위치 정보
  - name: 지역명(강남역, 정자동, 타임스퀘어)
  - relation: 관련성 (exact: 특정 식당명, nearby: 근처)
- category: 식당 카테고리(한식|일식|중식|양식|분식|퓨전요리|술집|카페|디저트)
- menu:
    - value: 메뉴명(예: 국밥, 치킨, 회, 돈가스, 파스타)
    - need_filter: 검색 할 때 필터를 해야 하는지 여부(1: 해야함, 0: 하지말아야 함)
- convenience:
    - value: 편의(예: 주차, 배달, 포장, 예약, 룸, 반려동물, 고기구워주는)
    - need_filter: 검색 할 때 필터를 해야 하는지 여부(1: 해야함, 0: 하지말아야 함)
- atmosphere: 분위기(예: 이국적인, 색다른, 로맨틱한, 조용한)
- occasion: 상황(예: 회식, 단체, 데이트, 혼밥, 가족)

응답 형식:
{{
  "location": [{{"name": "지역명", "relation": "exact|nearby"}}],
  "category": "식당 카테고리(한식|일식|중식|양식|분식|퓨전요리|술집|카페|디저트)",
  "menu": [{{"value": "메뉴명", "need_filter": 1|0}}],
  "convenience": [{{"value": "편의 사항", "need_filter": 1|0}}],
  "atmosphere": ["분위기"],
  "occasion": ["상황"]
}}

- 필터 해야하는 경우의 예시들(need_filter: 1)
ex1) query: "강남 주차되는 일식집" -> '주차'가 되는 식당을 검색하는 경우이기 때문에 필터 해야함
ex2) query: "정자역 오리 맛집" -> 정자역 주변에서 '오리' 메뉴를 판매하는 식당을 검색해야 하므로 필터 해야함
- 필터 하지 말아야 하는 경우의 예시들(need_filter: 0)
ex1) query: "버거킹 강남점 주차" -> 버거킹 강남점에 '주차'가 되는지를 묻는 것이기 때문에 먼저 버거킹 강남점을 필터없이 검색 후 주차 여부를 답변해주어야 하므로 필터 하지 말아야함
ex2) query: "판교 보노보노커피 메뉴" -> 판교 보노보노커피의 판매 메뉴를 묻고있는 것이므로 먼저 판교 보노보노커피를 필터없이 검색 후 메뉴 목록을 답변해야하므로 필터 하지 말아야함

예시:
1. "강남역 주차되는 일식집" -> {{"location": [{{"name": "강남역", "relation": "nearby"}}], "category": "일식", "convenience": [{{"value": "주차", "need_filter": 1}}]}}
2. "판교 애견동반 식당" -> {{"location": [{{"name": "판교", "relation": "nearby"}}], "convenience": [{{"value": "반려동물", "need_filter": 1}}]}}
3. "마포 진대감 주차되나요?" -> {{"location": [{{"name": "마포", "relation": "nearby"}}, {{"name": "진대감", "relation": "exact"}}], "convenience": [{{"value": "주차", "need_filter": 0}}]}}
4. "송파 이베리코 촌돼지 메뉴" -> {{"location": [{{"name": "송파", "relation": "nearby"}}, {{"name": "이베리코 촌돼지", "relation": "exact"}}],}}]
5. "서현 조용한 양갈비집" -> {{"location": [{{"name": "서현", "relation": "nearby"}}], "atmosphere": ["조용한"], "menu": [{{"value": "양갈비", "need_filter": 1}}]}}

실제 자연어 쿼리에 포함된 키워드에 관련한 것만 출력해주세요. 예측해서 추가하면 안됩니다.

자연어 쿼리: {query}
"""


class Location(BaseModel):
    name: str
    relation: str

class Menu(BaseModel):
    value: str
    need_filter: int

class Convenience(BaseModel):
    value: str
    need_filter: int


class StructuredQuery(BaseModel):
    location: list[Location]
    category: str 
    menu: list[Menu]
    convenience: list[Convenience] 
    atmosphere: list[str] 
    occasion: list 


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
            max_output_tokens=1024,
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
                "category": "일식",
                "convenience": [{"value": "주차", "need_filter": 1}]
            }
        },
        {
            "query": "판교 애견동반 식당",
            "expected": {
                "location": [{"name": "판교", "relation": "nearby"}],
                "convenience": [{"value": "반려동물", "need_filter": 1}]
            }
        },
        {
            "query": "마포 진대감 주차되나요?",
            "expected": {
                "location": [{"name": "마포", "relation": "nearby"}, {"name": "진대감", "relation": "exact"}],
                "convenience": [{"value": "주차", "need_filter": 0}]
            }
        },
        {
            "query": "조용히 대화할 수 있는 맥주집",
            "expected": {
                "category": "술집",
                "atmosphere": ["조용한"],
            }
        },
        {
            "query": "홍대에 회식하기 좋은 삼겹살집 추천해줘",
            "expected": {
                "location": [{"name": "홍대", "relation": "nearby"}],
                "menu": {"value": "삼겹살", "need_filter": 1},
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
