"""
식당 데이터를 검색용 문서로 변환하는 스크립트
README.md의 섹션 5에 따라 전처리 및 LLM 기반 정보 추출을 수행
"""

import os
import json
import re
import argparse
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


load_dotenv()

SYSTEM_PROMPT = """
당신은 주어진 식당 정보(소개글, 리뷰)에서 핵심 키워드를 정확하게 추출하는 맛집 데이터 분석가입니다.
추출된 정보는 식당 검색 시스템의 성능을 높이는 데 사용됩니다.
"""

EXTRACT_FEATURES_PROMPT = """
식당 소개글과 사용자 리뷰에서 아래 각 항목에 해당하는 특징 키워드가 있으면 추출해주세요.
1. `review_food`: 리뷰에서 언급된 메뉴나 음식 키워드  (예: 파스타, 스테이크, 떡볶이)
2. `convenience`: 식당에서 제공하는 긍정적인 편의 및 서비스 (예: 주차, 발렛, 배달, 포장, 예약, 룸, 콜키지, 반려동물, 와이파이, 24시, 구워줌)
3. `atmosphere`: 분위기 (예: 이국적인, 로맨틱한, 뷰맛집, 노포, 조용한, 시끌벅적한)
4. `occasion`: 방문 목적 (예: 데이트, 기념일, 회식, 단체, 혼밥, 혼술)
5. `features`: 기타 특징 (예: 넓은공간, 가성비)

**중요: 반드시 유효한 JSON 형식으로만 응답하세요. 추가적인 설명이나 마크다운은 포함하지 마세요.**

**추출 가이드라인:**
- 각 항목에 대해 10개 이하의 핵심 키워드를 추출합니다.
- 여러 리뷰에서 공통적으로 언급되는 내용을 우선적으로 고려합니다.
- '인생'이 들어가는 키워드는 추출하지 마세요.
- 절대로 같은 키워드를 중복해서 생성하지 마세요.
- 부정적인 내용은 편의 기능이 아님: '직접 구워먹어야 함'이나 '주차 공간 없음'과 같이 고객에게 불편을 주거나, 식당에서 제공하지 않는 서비스는 `convenience` 항목에 절대 포함하지 마세요.
- 리뷰나 소개글에 명시적으로 언급된 단어만으로 키워드를 추출하세요. 예를 들어 '전화하고 방문'을 '예약'으로 해석하는 것처럼 문맥을 확장하여 추론하지 마세요.
- 모든 키워드를 종합하여 중복을 제거해주세요.
- 항목에 키워드가 없다면 빈 배열 []을 반환하세요.
- 식당 소개글에서 메뉴는 제외 해주세요. 다른 특징만 추출하세요.

**JSON 형식:**
{{
    "review_food": list,
    "convenience": list,
    "atmosphere": list,
    "occasion": list,
    "features": list,
}}

위 가이드라인과 예시를 참고하여, 아래의 실제 입력 데이터에서 특징을 추출하세요.

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


_gemini_client = None
_openai_client = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client()
    return _gemini_client


def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


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


def extract_features_with_gemini(place_id: str, reviews: list[str], description: str) -> dict[str, list[str]]:
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

    response = get_gemini_client().models.generate_content(
        model="gemini-2.5-flash",
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=LLMFeatures,
            max_output_tokens=512,
            thinking_config=genai.types.ThinkingConfig(thinking_budget=0),
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
    

def extract_features_with_openai(place_id: str, reviews: list[str], description: str) -> dict[str, list[str]]:
    num_reviews_to_use = 30
    # 리뷰 텍스트 결합 (너무 길면 제한)
    review_text = "\n".join(reviews[:num_reviews_to_use])  # 최대 30개 리뷰만 사용
    if len(review_text) > 100 * num_reviews_to_use:
        review_text = review_text[:100 * num_reviews_to_use]

    user_prompt = EXTRACT_FEATURES_PROMPT.format(description=description, reviews=review_text)

    response = get_openai_client().responses.parse(
        model="gpt-5-mini",
        input=[
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user_prompt },
        ],
        text_format=LLMFeatures,
    )

    try:
        llm_features = response.output_parsed.model_dump()
    except:
        print(f"JSONDecodeError: {place_id}")
        return None

    return llm_features

def extract_features(
    place_id: str,
    reviews: list[str],
    description: str,
    platform: str = 'openai',
) -> dict[str, list[str]]:
    if platform == "gemini":
        return extract_features_with_gemini(place_id, reviews, description)
    
    return extract_features_with_openai(place_id, reviews, description)
    


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



def process_restaurant(raw_data: dict[str, Any], platform: str = 'openai') -> dict[str, Any]:
    """
    원본 식당 데이터를 검색용 문서로 변환
    """
    # 1. 전처리
    # 카테고리 클리닝

    # processed_category = convert_category(raw_data["category"])

    # 가격 변환
    processed_menus = []
    for menu in raw_data.get("menus", []):
        processed_menu = menu.copy()
        processed_menu["price"] = convert_price_to_int(processed_menu["price"])
        processed_menus.append(processed_menu)
    
    # 위도, 경도 변환
    lat, lon = convert_coordinates(raw_data.get("mapx"), raw_data.get("mapy"))
    
    # 2. LLM을 사용한 특징 추출
    reviews = [review.replace("\n", " ").strip() for review in raw_data.get("reviews", []) if review.replace("\n", " ").strip()]
    reviews = [review for review in raw_data.get("reviews", []) if len(review) >= 15]
    extracted_features = extract_features(
        raw_data["place_id"],
        reviews,
        raw_data.get("description", ""),
        platform
    )

    # 3. 요약 생성
    summary = create_summary(
        title=raw_data.get("title", ""),
        category=raw_data.get("category"),
        address=raw_data.get("address", ""),
        road_address=raw_data.get("roadAddress", ""),
        menus=processed_menus,
        review_food=extracted_features.get("review_food"),
        convenience=extracted_features.get("convenience"),
        atmosphere=extracted_features.get("atmosphere"),
        occasion=extracted_features.get("occasion"),
        features=extracted_features.get("features")
    )
    
    # 6. 최종 문서 생성
    document = {
        "place_id": raw_data.get("place_id"),
        "title": raw_data.get("title"),
        "category": raw_data.get("category"),
        "address": raw_data.get("address"),
        "roadAddress": raw_data.get("roadAddress"),
        "coordinate": {
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


def process_single_restaurant(args_tuple):
    """단일 식당 데이터 처리 (병렬 처리용)"""
    raw_data, platform = args_tuple
    try:
        return process_restaurant(raw_data, platform)
    except Exception as e:
        print(f"Error processing {raw_data.get('place_id', 'unknown')}: {e}")
        return None


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "../../data/crawled_restaurants")
    OUTPUT_DIR = os.path.join(BASE_DIR, "../../data/featured_restaurants")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 입력 디렉토리에서 모든 part 파일 찾기
    input_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("part-") and f.endswith(".jsonl")]
    input_files.sort()  # 파일명 순서로 정렬
    
    print(f"📁 총 {len(input_files)}개 파일 처리 시작\n")
    
    for file_idx, input_filename in enumerate(input_files, 1):
        input_file_path = os.path.join(INPUT_DIR, input_filename)
        output_file_path = os.path.join(OUTPUT_DIR, input_filename)
        
        print(f"📄 [{file_idx}/{len(input_files)}] {input_filename}")
        
        # 전체 레코드 수 계산
        total_records = 0
        with open(input_file_path, "r", encoding="utf-8") as f_in:
            for _ in f_in:
                total_records += 1
        
        # 이미 처리된 place_id들 확인
        processed_place_ids = set()
        if os.path.exists(output_file_path):
            with open(output_file_path, "r", encoding="utf-8") as f_out:
                for line in f_out:
                    document = json.loads(line)
                    processed_place_ids.add(document["place_id"])
        
        processed_count = len(processed_place_ids)
        remaining_count = total_records - processed_count
        
        print(f"전체: {total_records}개 | 완료: {processed_count}개 | 남은작업: {remaining_count}개")
        
        if remaining_count == 0:
            print("✅ 이미 모든 데이터 처리 완료\n")
            continue
        
        # 파일 처리 (병렬 처리)
        failed_count = 0
        
        # 처리할 데이터 수집
        tasks = []
        with open(input_file_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                raw_data = json.loads(line)
                if raw_data["place_id"] not in processed_place_ids:
                    tasks.append((raw_data, platform))
        
        # 병렬 처리 실행
        with open(output_file_path, "a", encoding="utf-8") as f_out:
            progress_bar = tqdm(
                total=len(tasks),
                desc="처리중",
                unit="개",
                ncols=80,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            with ThreadPoolExecutor(max_workers=parallelism) as executor:
                # 작업 제출
                future_to_task = {executor.submit(process_single_restaurant, task): task for task in tasks}
                
                # 완료된 작업 처리
                for future in as_completed(future_to_task):
                    document = future.result()
                    if not document:
                        failed_count += 1
                    else:
                        f_out.write(f"{json.dumps(document, ensure_ascii=False)}\n")
                        f_out.flush()  # 즉시 파일에 쓰기
                    
                    progress_bar.set_postfix({"실패": failed_count})
                    progress_bar.update(1)
            
            progress_bar.close()
        
        print(f"✅ 완료 - 실패: {failed_count}개\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="식당 데이터에서 특징을 추출하는 스크립트")
    parser.add_argument(
        "--platform", 
        choices=["openai", "gemini"], 
        default="openai",
        help="사용할 LLM 플랫폼 (기본값: openai)"
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=5,
        help="동시 처리할 요청 수 (기본값: 5)"
    )
    
    args = parser.parse_args()
    platform = args.platform
    parallelism = args.parallelism
    
    print(f"🤖 사용 플랫폼: {platform}")
    print(f"🔄 병렬 처리: {parallelism}개 동시 요청\n")
    
    main()
