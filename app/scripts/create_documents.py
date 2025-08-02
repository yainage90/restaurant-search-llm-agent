"""
식당 데이터를 검색용 문서로 변환하는 스크립트
README.md의 섹션 5에 따라 전처리 및 LLM 기반 정보 추출을 수행
"""

import os
import json
import re
from typing import Any
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from app.retrieve.embeddings import get_document_embeddings


load_dotenv()

SYSTEM_PROMPT = """
당신은 주어진 식당 정보(소개글, 리뷰)를 분석하여, 사용자들이 검색할 때 사용할 만한 핵심 특징들을 정확하게 추출하는 맛집 데이터 분석가입니다.
추출된 정보는 식당 검색 시스템의 성능을 높이는 데 사용됩니다.
"""

EXTRACT_FEATURES_PROMPT = """
식당 소개글과 사용자 리뷰를 바탕으로, 아래 각 항목에 해당하는 특징 키워드를 추출해주세요.

**추출 가이드라인:**
- 각 항목에 대해 10개 이하의 핵심 키워드를 추출합니다.
- 키워드는 명사 형태로 간결하게 표현해주세요. (예: "주차 가능" -> "주차")
- 직접적으로 등장하는 표현이 아니면 추출하지마세요.
- 여러 리뷰에서 공통적으로 언급되는 내용을 우선적으로 고려합니다.
- 모든 키워드를 종합하여 중복을 제거해주세요.
- 최종 결과는 반드시 JSON 형식으로 반환해야 합니다.

**추출 항목:**
{{
    "review_food": "리뷰에서 언급된 메뉴나 음식 키워드 (예: 파스타, 스테이크, 떡볶이)",
    "convenience": "편의 시설 및 서비스 (예: 주차, 발렛, 배달, 포장, 예약, 룸, 콜키지, 반려동물, 와이파이)",
    "atmosphere": "식당의 전반적인 분위기 (예: 이국적인, 로맨틱한, 뷰맛집, 노포, 조용한, 시끌벅적한)",
    "occasion": "어떤 상황에 어울리는지 (예: 데이트, 기념일, 회식, 단체, 가족, 혼밥, 모임)",
    "features": "기타 특징 (예: 유명인 방문 - 유명인 이름, 넓은공간, 가성비, 친절한, 웨이팅)"
}}

---
**입력 예시:**

식당 소개글:
"강남역 최고의 이탈리안 레스토랑, '파스타리오'입니다. 장인이 직접 뽑은 생면으로 만든 파스타와 참나무 화덕에서 구운 피자가 일품입니다. 기념일을 위한 로맨틱한 창가 자리가 마련되어 있으며, 단체 회식을 위한 별도의 룸도 완비하고 있습니다. 발렛 주차 서비스를 제공하여 편리하게 방문하실 수 있습니다."

사용자 리뷰:
- "파스타가 정말 인생 파스타였어요! 특히 크림 파스타 추천합니다. 분위기가 좋아서 데이트 장소로 딱이에요."
- "창가 자리에 앉았는데 뷰가 너무 좋았어요. 소개팅했는데 성공적이었습니다. 주차도 발렛이 돼서 편했어요."
- "팀 회식으로 다녀왔는데, 룸이 있어서 우리끼리 편하게 즐길 수 있었어요. 양도 많고 가성비가 좋네요."

**출력 예시:**
{{
    "review_food": ["파스타", "크림 파스타", "피자"],
    "convenience": ["주차", "발렛", "예약", "룸"],
    "atmosphere": ["로맨틱한", "뷰맛집"],
    "occasion": ["데이트", "기념일", "회식", "단체", "소개팅"],
    "features": ["생면", "화덕피자", "창가자리", "가성비"]
}}
---

**실제 입력:**

식당 소개글:
{description}

사용자 리뷰:
{reviews}
"""


class LLMFeatures(BaseModel):
    review_food: list[str]
    convenience: list[str]
    atmosphere: list[str]
    occasion: list[str]
    features: list[str]


llm = genai.Client()

def convert_price_to_int(price_str: str) -> int | None:
    """
    가격 문자열을 정수로 변환
    예: "20,000원" -> 20000
    """
    if not price_str:
        return None
    
    # 숫자와 콤마만 추출
    numbers = re.findall(r'[\d,]+', price_str)
    if not numbers:
        return None
    
    # 콤마 제거하고 정수로 변환
    try:
        return int(numbers[0].replace(',', ''))
    except ValueError:
        return None


def convert_coordinates(mapx: str, mapy: str) -> tuple[float, float]:
    """
    네이버 맵 좌표를 위도, 경도로 변환
    mapx: 앞 세 자리가 정수부, 나머지가 소수부 (예: "1271551201" -> 127.1551201)
    mapy: 앞 두 자리가 정수부, 나머지가 소수부 (예: "375630641" -> 37.5630641)
    """

    lat = float(mapy[:2] + '.' + mapy[2:])
    lon = float(mapx[:3] + '.' + mapx[3:])
    
    return lat, lon


def extract_features_with_llm(reviews: list[str], description: str) -> dict[str, list[str]]:
    """
    LLM을 사용하여 리뷰와 설명에서 특징을 추출
    실제 구현시에는 OpenAI API 등을 사용
    """

    num_reviews_to_use = 30
    # 리뷰 텍스트 결합 (너무 길면 제한)
    review_text = "\n".join(reviews[:num_reviews_to_use])  # 최대 30개 리뷰만 사용
    if len(review_text) > 100 * num_reviews_to_use:
        review_text = review_text[:100 * num_reviews_to_use]

    user_prompt = EXTRACT_FEATURES_PROMPT.format(description=description, reviews=review_text)

    response = llm.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=user_prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "response_mime_type": "application/json",
            "response_schema": LLMFeatures,
            "temperature": 0.0,
        }
        # config=genai.types.GenerateContentConfig(
        #     temperature=0.0,
        #     system_instruction=SYSTEM_PROMPT,
        #     response_mime_type="application/json",
        #     response_schema=LLMFeatures,
        # )
    )

    response_text = response.text
    # LLM 응답에 포함된 마크다운을 제거
    if "```" in response_text:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            response_text = match.group(0)

    llm_features = json.loads(response_text)

    return llm_features
    

def create_summary(
    title: str,
    address: str,
    road_address: str,
    menus: list[dict], 
    review_food: list[str] | None,
    convenience: list[str] | None,
    atmosphere: list[str] | None, 
    occasion: list[str] | None,
    features: list[str] | None,
) -> str:
    """
    임베딩을 위한 요약 텍스트 생성
    """
    # 메뉴명 추출
    menu_names = [menu.get("name", "") for menu in menus if menu.get("name")]
    all_menus = menu_names + (review_food or [])
    
    summary_parts = [
        f"식당 이름: {title}",
        f"주소: {address}({road_address})",
        f"메뉴: {','.join(all_menus)}" if all_menus else "메뉴: 정보 없음"
    ]
    
    if convenience:
        summary_parts.append(f"편의: {','.join(convenience)}")
    
    if atmosphere:
        summary_parts.append(f"분위기: {','.join(atmosphere)}")
    
    if occasion:
        summary_parts.append(f"상황: {','.join(occasion)}")
    
    if features:
        summary_parts.append(f"기타 특징: {','.join(features)}")
    
    return "\n".join(summary_parts)



def process_restaurant(raw_data: dict[str, Any]) -> dict[str, Any]:
    """
    원본 식당 데이터를 검색용 문서로 변환
    """
    # 1. 전처리
    # 가격 변환
    processed_menus = []
    for menu in raw_data.get("menus", []):
        processed_menu = menu.copy()
        processed_menu["price"] = convert_price_to_int(processed_menu["price"])
        processed_menus.append(processed_menu)
    
    # 위도, 경도 변환
    lat, lon = convert_coordinates(raw_data.get("mapx"), raw_data.get("mapy"))
    
    # 2. LLM을 사용한 특징 추출
    extracted_features = extract_features_with_llm(
        raw_data.get("reviews", []),
        raw_data.get("description", "")
    )
    
    # 3. 요약 생성
    summary = create_summary(
        raw_data.get("title", ""),
        raw_data.get("address", ""),
        raw_data.get("roadAddress", ""),
        processed_menus,
        extracted_features.get("review_food"),
        extracted_features.get("convenience"),
        extracted_features.get("atmosphere"),
        extracted_features.get("occasion"),
        extracted_features.get("features")
    )
    
    # 4. 임베딩 추출
    embedding = get_document_embeddings(summary)[0]
    
    # 6. 최종 문서 생성
    document = {
        "place_id": raw_data.get("place_id"),
        "title": raw_data.get("title"),
        "address": raw_data.get("address"),
        "roadAddress": raw_data.get("roadAddress"),
        "location": {
            "lat": lat,
            "lon": lon,
        },
        "menus": processed_menus,
        "reviews": raw_data.get("reviews", []),
        "description": raw_data.get("description", ""),
        "review_food": extracted_features.get("review_food"),
        "convenience": extracted_features.get("convenience"),
        "atmosphere": extracted_features.get("atmosphere"),
        "occasion": extracted_features.get("occasion"),
        "features": extracted_features.get("features"),
        "summary": summary,
        "embedding": embedding,
    }
    
    return document


def print_test():
    """
    메인 처리 함수
    """
    # 예시 데이터 처리
    sample_data = {
        "place_id": "1993900101",
        "title": "우드멜로우",
        "address": "서울특별시 강동구 고덕동 482",
        "roadAddress": "서울특별시 강동구 아리수로 243",
        "mapx": "1271551201",
        "mapy": "375630641",
        "menus": [
            {"name": "멜란자네파다노", "price": "20,000원"},
            {"name": "냉파스타(여름시즌한정)", "price": "17,500원"},
        ],
        "reviews": [
            "우드멜로우에서 브런치를 즐기고 왔어요. 긴 테이블이 있어서 부모님과 여럿이 함께 앉기 편했어요.",
            "브런치 먹을만한 곳을 찾다가 오픈시간도 9시 30분이고 거리도 가까워서 와봤는데 브런치 메뉴가 정말 맛있네요"
        ],
        "description": "예쁘고 편안한 공간에서 맛있는 커피와 브런치를 즐겨보세요 :)"
    }
    
    # 문서 처리
    document = process_restaurant(sample_data)
    
    # 결과 출력
    print(json.dumps(document, ensure_ascii=False, indent=2))


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(BASE_DIR, "../../data/crawled_restaurants.jsonl")
    OUPUT_FILE = os.path.join(BASE_DIR, "../../data/documents.jsonl")

    processed_place_ids = set()
    if os.path.exists(OUPUT_FILE):
        with open(OUPUT_FILE, "r", encoding="utf-8") as f_out:
            for line in f_out:
                document = json.loads(line)
                processed_place_ids.add(document["place_id"])

    with open(OUPUT_FILE, "a", encoding="utf-8") as f_out:
        with open(INPUT_FILE, "r", encoding="utf-8") as f_in:
            for line in f_in:
                raw_data = json.loads(line)
                if raw_data["place_id"] in processed_place_ids:
                    continue

                document = process_restaurant(raw_data)
                f_out.write(f"{json.dumps(document, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()