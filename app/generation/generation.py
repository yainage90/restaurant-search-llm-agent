from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()


SYSTEM_PROMPT = """당신은 한국의 식당 정보를 제공하는 전문 AI 어시스턴트입니다.
제공된 식당 정보를 바탕으로 사용자의 질문에 정확하고 유용한 답변을 제공해주세요.

다음 지침을 따라주세요:
1. 제공된 식당 정보만을 사용하여 답변하세요.
2. 메뉴, 가격, 위치, 분위기, 편의시설 등 구체적인 정보를 포함해주세요.
3. 사용자가 식당을 선택하는데 도움이 되도록 상세하고 친절하게 설명해주세요.
4. 정보가 부족한 경우 솔직하게 말씀드리세요.
5. 한국어로 자연스럽게 대화하듯 답변해주세요."""

GENERATION_PROMPT = """사용자 질문: {query}

식당 정보:
{context}

위 식당 정보를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요."""

def generate(query: str, context: str) -> str:
    """
    사용자 쿼리와 검색된 컨텍스트를 받아서 LLM에 응답 생성을 요청합니다.
    
    Args:
        query: 사용자의 자연어 질문
        context: 검색된 식당 정보 문서들
        
    Returns:
        LLM이 생성한 응답 텍스트
        
    Raises:
        Exception: LLM 호출 중 오류가 발생한 경우
    """

    # 프롬프트 생성
    if query:
        user_prompt = GENERATION_PROMPT.format(query=query, context=context)
        
        # LLM에 요청
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1024,
            )
        )
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=context,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=1024,
            )
                
        )

    generated_text = response.text


    return generated_text


def generate_streaming(chat_history: str):
    """
    사용자 쿼리와 검색된 컨텍스트를 받아서 LLM에 스트리밍 응답 생성을 요청합니다.
    
    Args:
        context: 검색된 식당 정보 문서들 (빈 문자열이면 일반 대화)
        
    Yields:
        str: 스트리밍되는 응답 청크
    """
    
    # LLM에 스트리밍 요청
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash-lite",
        contents=chat_history,
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=1024,
        ),
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text
        

def test_generation():
    query = "강남역 근처에서 주차 가능한 일식집 추천해주세요"
    
    context = """문서1:
식당 이름: 스시젠 강남점
주소: 서울특별시 강남구 역삼동 123-45 (강남대로 456)
메뉴: 오마카세(80000원), 연어사시미(25000원), 참치사시미(30000원), 우니초밥(15000원), 장어덮밥(18000원)
편의: 주차, 예약, 포장, 발렛파킹
분위기: 고급스러운, 조용한, 모던한
상황: 데이트, 회식, 기념일, 접대
기타 특징: 셰프 추천 코스, 신선한 재료, 개별룸, 카운터석

문서2:
식당 이름: 도쿄스시 강남본점  
주소: 서울특별시 강남구 논현동 789-12 (테헤란로 321)
메뉴: 특선초밥세트(45000원), 회덮밥(22000원), 새우튀김(18000원), 미소시루(8000원), 사케(12000원)
편의: 주차, 포장, 배달
분위기: 캐주얼한, 가족적인
상황: 가족식사, 점심, 혼밥
기타 특징: 가성비, 빠른 서비스, 런치세트 할인

문서3:
식당 이름: 이자카야 하나
주소: 서울특별시 강남구 신사동 456-78 (강남대로 654)  
메뉴: 사시미모둠(55000원), 야키토리(8000원), 하이볼(9000원), 사케(15000원), 라멘(14000원)
편의: 주차, 예약
분위기: 이국적인, 활기찬, 술집같은
상황: 회식, 술자리, 친구모임
기타 특징: 일본 현지 맛, 다양한 사케, 늦은 시간 영업"""

    try:
        generated_text = generate(query, context)
        print(f"query: {query}")
        print(f"context: \n{context}")
        print("=== 생성된 응답 ===")
        print(generated_text)
    except Exception as e:
        print(f"테스트 실행 중 오류 발생: {e}")


if __name__ == "__main__":
    test_generation()
