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
1. `review_food`: 리뷰에서 언급된 메뉴나 음식 키워드  (예: 파스타, 스테이크, 떡볶이)
2. `convenience`: 편의 및 서비스 (예: 주차, 발렛, 배달, 포장, 예약, 룸, 콜키지, 반려동물, 와이파이, 24시, 구워줌)
3. `atmosphere`: 분위기 (예: 이국적인, 로맨틱한, 뷰맛집, 노포, 조용한, 시끌벅적한)
4. `occasion`: 어떤 상황에 어울리는지 (예: 데이트, 기념일, 회식, 단체, 가족, 혼밥, 모임, 혼술)
5. `features`: 기타 특징 (예: 유명인 방문 - 유명인 이름, 넓은공간, 가성비)

**중요: 반드시 유효한 JSON 형식으로만 응답하세요. 추가적인 설명이나 마크다운은 포함하지 마세요.**

**추출 가이드라인:**
- 각 항목에 대해 10개 이하의 핵심 키워드를 추출합니다.
- 여러 리뷰에서 공통적으로 언급되는 내용을 우선적으로 고려합니다.
- '인생'이 들어가는 키워드는 절대로 추출하지 마세요.
- 절대로 같은 키워드를 중복해서 생성하지 마세요.
- 모든 키워드를 종합하여 중복을 제거해주세요.
- 키워드가 없다면 빈 배열 []을 반환하세요.
- 반드시 리뷰 본문에 명시적으로 언급된 단어들 중에서만 키워드를 추출해야 합니다. 본문에 없는 내용은 절대 포함해서는 안 됩니다.
- 키워드는 명사 형태로 간결하게 표현해주세요.(ex. 주차 가능 -> 주차)
- 식당 소개글에서는 메뉴는 추출하지 마세요. 다른 중요한 특징만 추출하세요.

**JSON 형식:**
{{
    "review_food": list,
    "convenience": list,
    "atmosphere": list,
    "occasion": list,
    "features": list,
}}

---
**입력 예시 1:**

식당 소개글:
"강남역 최고의 이탈리안 레스토랑, '파스타리오'입니다. 장인이 직접 뽑은 생면으로 만든 파스타와 참나무 화덕에서 구운 피자가 일품입니다. 기념일을 위한 로맨틱한 창가 자리가 마련되어 있으며, 단체 회식을 위한 별도의 룸도 완비하고 있습니다. 발렛파킹 서비스를 제공하여 편리하게 방문하실 수 있습니다."

사용자 리뷰:
- "파스타가 정말 인생 파스타였어요! 특히 크림 파스타 추천합니다. 분위기가 좋아서 데이트 장소로 딱이에요."
- "창가 자리에 앉았는데 뷰가 너무 좋았어요. 소개팅했는데 성공적이었습니다. 주차도 발렛이 돼서 편했어요."
- "팀 회식으로 다녀왔는데, 룸이 있어서 우리끼리 편하게 즐길 수 있었어요. 양도 많고 가성비가 좋네요."

**출력 예시 1:**
{{
    "review_food": ["파스타", "크림 파스타"],
    "convenience": ["발렛", "예약", "룸"],
    "atmosphere": ["로맨틱한"],
    "occasion": ["데이트", "기념일", "회식", "소개팅"],
    "features": ["창가자리", "가성비"]
}}
---
**입력 예시 2:**

식당 소개글:
"도마우에는 휴양지 컨셉의 맥주.위스키.와인.사케.등 다양한 술을 마실수 있는 펍입니다.처음처럼,참이슬,진로이즈벡 안팔아요 단 소주는 화요,일품진로,아와모리 잔파는 판매하고 있습니다.주차공간을 따로 보유하고 있지는 않습니다."

사용자 리뷰:
- 혼술 좋아하는 분들에게 너무 좋은 곳이에요 음식도 다 너무 맛있습니다. 개인적으로 열빙어 튀김이 진짜 맛있었던것 같습니다 튀김 진짜 잘하세요🍤🍤\n\n무엇보다 오키나와 생맥주를 마실 수 있는 점이 진짜 좋았습니다ㅜㅜ 생맥 너무 맛있어서 한번 더 시켰어요"
- 분위기 좋은 하계역 술집이라해서 방문했어요\n• 아늑한 분위기의 이자카야\n• 대화하기 좋은 조용한 하계동 술집\n\n오키나와 생맥주가 파는게 독특해서 좋아요

**출력 예시 2:**
{{
    "review_food": ["열빙어튀김", "오키나와 생맥주"],
    "occasion": ["혼술"],
    "atmosphere": ["조용한"],
    "features": ["휴양지 컨셉"]
}}
---
**입력 예시 3:**

식당 소개글:
""

사용자 리뷰:
- 구워줘서 아주 좋아요
- 넘 맛난것^^ 소금구이 완전 꼬숩고 담백함\n조만간 다시 방문할듯!!
- 답십리에서 유명한 민물장어\n동네 맛집 입니다\n주문 즉시 사장님께서 직접 구워서\n먹기 편하게 내어주시니 완전 좋습니다
- 초복기념으로 가족 식사했어요\n가게가 큰편은 아니라 오래 기다렸지만 맛있었어요
- 사장님이 친절해요
- 초복기념으로 가족 식사했어요\n가게가 큰편은 아니라 오래 기다렸지만 맛있었어요

**출력 예시 3:**
{{
    "review_food": ["민물장어", "소금구이"],
    "convenience": ["구워줌"],
    "occasion": ["가족"],
    "features": ["초복", "친절"]
}}
---
**입력 예시 4:**

식당 소개글:
"화 - 일 11:00 ~ 21:00\n월요일 휴무\n\n칠곡호수를 품은 Cafe Mell-Mell은\n묵직한 바디감과 고소한 풍미가 특징인 멜멜 시그니처 원두를 사용하며,\n디카페인 원두도 준비되어 있습니다 :)"

사용자 리뷰:
매장이 넓고 뷰가 좋아요
밀크초코 크로와상 너무 맛있어용 ㅠㅅㅠ 프레즐도 맛있었음 다 먹어보지는 못했지만 요기는 전체적으로 빵이 맛있나봐여 그거에 비해 커피는 그냥 그랬음 내부 엄청 넓고 주차장도 잘 되어 있고 무엇보다 뷰가 너무 좋음!
칠곡저수지에 오면 꼭 오는 멋진 카페입니다.\n오늘은 날씨도 너무 좋고 커피도 너무 맛있어요~
뷰가 1층부터 3층까지 통창이라 너무 이쁘고 인테리어도 너무좋아요 주차공간도 넓고 몰랐던 곳인데 알게되어서 좋습니다 음료도 맛있어요!!
저수지뷰라고해서 와봤어요 ㅎㅎ 매장도 너무 넓고 일단 인테리어가 완전 mz감성!!!!! 포토존도 따로되어있고 직원분들도 친절하시네요 ㅎㅎ빵도 생각보다 다양하고 커피도 가격이 착하네용 ㅎㅎ 단골될것같아요!!
처음 방문했는데 주차장도 되게 넓고 뷰가 예뻐요!
대추차 너무 진하고 좋아요\n커피는 그냥그래요\n분위기는 좋아요


**출력 예시 4:**
{{
    "review_food": ["밀크초코 크로와상", "프레즐", "빵", "대추차"], 
    "convenience": ["주차"],
    "atmosphere": ["뷰맛집", "저수지뷰"],
    "features": ["월요일 휴무", "칠곡호수", "넓은 매장"]
}}


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


def convert_category(category: str) -> str:
    categories = [c.strip() for c in category.split(">")]
    if categories[0] != "음식점":
        category = categories[0]
    elif len(categories) > 1:
        category = categories[1]

    if "," in category:
        category = [c.strip() for c in category.split(",")]
    else:
        category = [category]

    return category
    

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


def extract_features_with_llm(place_id: str, reviews: list[str], description: str) -> dict[str, list[str]]:
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
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LLMFeatures,
            max_output_tokens=1024,
        )
    )

    response_text = response.text
        
    # LLM 응답에 포함된 마크다운을 제거
    if "```" in response_text:
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if match:
            response_text = match.group(0)

    try:
        llm_features = json.loads(response_text)
    except:
        print(f"JSONDecodeError: {place_id}")
        return None

    return llm_features
    

def create_summary(
    title: str,
    category: str,
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
    menus = [f"{menu.get("name", "")}({menu.get("price", "N/A")}원)" for menu in menus if menu.get("name")]
    all_menus = menus + (review_food or [])
    
    summary_parts = [
        f"식당 이름: {title}",
        f"카테고리: {category}",
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
    # 카테고리 클리닝

    processed_category = convert_category(raw_data["category"])

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
        raw_data["place_id"],
        raw_data.get("reviews", []),
        raw_data.get("description", "")
    )

    if not extracted_features:
        return None
    
    # 3. 요약 생성
    summary = create_summary(
        raw_data.get("title", ""),
        raw_data.get("category"),
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
        "category": processed_category,
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
    INPUT_DIR = os.path.join(BASE_DIR, "../../data/crawled_restaurants")
    OUTPUT_DIR = os.path.join(BASE_DIR, "../../data/documents")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 입력 디렉토리에서 모든 part 파일 찾기
    input_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("part-") and f.endswith(".jsonl")]
    input_files.sort()  # 파일명 순서로 정렬
    
    for input_filename in input_files:
        input_file_path = os.path.join(INPUT_DIR, input_filename)
        output_file_path = os.path.join(OUTPUT_DIR, input_filename)
        
        print(f"Processing {input_filename}...")
        
        # 이미 처리된 place_id들 확인
        processed_place_ids = set()
        if os.path.exists(output_file_path):
            with open(output_file_path, "r", encoding="utf-8") as f_out:
                for line in f_out:
                    document = json.loads(line)
                    processed_place_ids.add(document["place_id"])
        
        # 파일 처리
        with open(output_file_path, "a", encoding="utf-8") as f_out:
            with open(input_file_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    raw_data = json.loads(line)
                    if raw_data["place_id"] in processed_place_ids:
                        continue

                    document = process_restaurant(raw_data)
                    if not document:
                        continue

                    f_out.write(f"{json.dumps(document, ensure_ascii=False)}\n")
        
        print(f"Completed {input_filename}")


if __name__ == "__main__":
    main()
