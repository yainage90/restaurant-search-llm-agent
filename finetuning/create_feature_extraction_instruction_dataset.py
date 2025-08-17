# -*- coding: utf-8 -*-
"""
특징 추출 경량 모델 파인튜닝을 위한 instruction dataset 생성 스크립트
data/featured_restaurants의 데이터를 기반으로 instruction-output 쌍을 생성
"""

import os
import json
import argparse
from typing import Any
from tqdm import tqdm


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


def create_instruction_dataset_entry(featured_data: dict[str, Any]) -> dict[str, Any]:
    """
    featured_restaurants 데이터를 instruction dataset 형태로 변환
    """
    # 리뷰 텍스트 결합 (최대 30개 리뷰만 사용)
    num_reviews_to_use = 30
    reviews = featured_data.get("reviews", [])
    reviews = [review for review in reviews if len(review) >= 15]
    review_text = "\n".join(reviews[:num_reviews_to_use])
    if len(review_text) > 100 * num_reviews_to_use:
        review_text = review_text[:100 * num_reviews_to_use]
    
    # 입력 프롬프트 생성
    user_prompt = EXTRACT_FEATURES_PROMPT.format(
        description=featured_data.get("description", ""),
        reviews=review_text
    )
    
    # 출력 JSON 생성
    output_json = {
        "review_food": featured_data.get("review_food", []),
        "convenience": featured_data.get("convenience", []),
        "atmosphere": featured_data.get("atmosphere", []),
        "occasion": featured_data.get("occasion", []),
        "features": featured_data.get("features", [])
    }
    
    # instruction dataset 형태로 변환
    instruction_entry = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user", 
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": json.dumps(output_json, ensure_ascii=False)
            }
        ]
    }
    
    return instruction_entry


def process_file(input_file_path: str, output_file_path: str, max_samples: int | None = None):
    """
    단일 파일을 처리하여 instruction dataset으로 변환
    """
    print(f"📄 처리 중: {os.path.basename(input_file_path)}")
    
    # 전체 레코드 수 계산
    total_records = 0
    with open(input_file_path, "r", encoding="utf-8") as f_in:
        for _ in f_in:
            total_records += 1
    
    # max_samples 제한이 있는 경우 적용
    target_samples = min(max_samples, total_records) if max_samples else total_records
    
    print(f"전체: {total_records}개 | 목표: {target_samples}개")
    
    # 파일 처리
    with open(input_file_path, "r", encoding="utf-8") as f_in:
        with open(output_file_path, "a", encoding="utf-8") as f_out:
            progress_bar = tqdm(
                total=total_records,
                desc="변환중",
                unit="개",
                ncols=80,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
            
            samples_added = 0
            for line in f_in:
                featured_data = json.loads(line)
                
                instruction_entry = create_instruction_dataset_entry(featured_data)
                f_out.write(f"{json.dumps(instruction_entry, ensure_ascii=False)}\n")
                f_out.flush()
                
                samples_added += 1
                progress_bar.update(1)
                    
            progress_bar.close()
    
    print(f"✅ 완료 - {samples_added}개 변환됨\n")


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "../data/featured_restaurants")
    OUTPUT_DIR = os.path.join(BASE_DIR, "../data/feature_extraction_instruction_dataset")
    
    # 출력 디렉토리 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 입력 디렉토리에서 모든 part 파일 찾기
    input_files = [f for f in os.listdir(INPUT_DIR) if f.startswith("part-") and f.endswith(".jsonl")]
    input_files.sort()
    
    print(f"📁 총 {len(input_files)}개 파일 처리 시작")
    print(f"🎯 각 파일당 최대 샘플 수: {args.max_samples_per_file if args.max_samples_per_file else '제한없음'}")
    print()
    
    total_samples = 0
    
    for file_idx, input_filename in enumerate(input_files, 1):
        input_file_path = os.path.join(INPUT_DIR, input_filename)
        output_file_path = os.path.join(OUTPUT_DIR, input_filename)
        
        print(f"[{file_idx}/{len(input_files)}]", end=" ")
        
        # 파일별 최대 샘플 수 제한이 있는 경우만 처리
        if args.max_files and file_idx > args.max_files:
            break
            
        process_file(input_file_path, output_file_path, args.max_samples_per_file)
    
    # 최종 통계
    print("🎉 전체 처리 완료!")
    total_samples = 0
    for filename in os.listdir(OUTPUT_DIR):
        if filename.startswith("part-") and filename.endswith(".jsonl"):
            with open(os.path.join(OUTPUT_DIR, filename), "r", encoding="utf-8") as f:
                total_samples += sum(1 for _ in f)
    
    print(f"📊 총 생성된 instruction 샘플 수: {total_samples:,}개")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="특징 추출 instruction dataset 생성 스크립트")
    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=None,
        help="각 파일당 최대 샘플 수 (기본값: 제한없음)"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="처리할 최대 파일 수 (기본값: 전체)"
    )
    
    args = parser.parse_args()
    
    main()