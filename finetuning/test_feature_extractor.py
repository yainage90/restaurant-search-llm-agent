#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
파인튜닝된 Gemma 3 1B IT 모델을 사용한 추론 테스트 스크립트

실행 방법:
    export HF_TOKEN=<huggingface access token>
    uv run python finetuning/inference.py
"""

import os
import json
import random
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

load_dotenv()

def load_fine_tuned_model(adapter_path: str):
    """파인튜닝된 모델과 토크나이저 로드"""
    
    # 베이스 모델 설정
    base_model_id = "google/gemma-3-1b-it"
    
    # QLoRA 설정 (추론용)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 베이스 모델 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    
    # PEFT 어댑터 로드
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print(f"device: {model.device}")
    
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str) -> str:
    """모델을 사용하여 응답 생성"""
    
    # 메시지 형태로 변환
    messages = [
        {
            "role": "system",
            "content": """당신은 주어진 식당 정보(소개글, 리뷰)에서 핵심 키워드를 정확하게 추출하는 맛집 데이터 분석가입니다.
추출된 정보는 식당 검색 시스템의 성능을 높이는 데 사용됩니다."""
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # 채팅 템플릿 적용
    chat_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 토크나이징
    inputs = tokenizer(
        chat_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(model.device)
    
    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # 응답 디코딩 (입력 프롬프트 제외)
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:], 
        skip_special_tokens=True
    )
    
    return response.strip()


def load_random_restaurant_data():
    """data/crawled_restaurants/part-00030.jsonl에서 랜덤한 레스토랑 데이터 로드"""
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "../data/crawled_restaurants/part-00030.jsonl")
    
    if not os.path.exists(data_path):
        print(f"❌ 데이터 파일이 존재하지 않습니다: {data_path}")
        return None
    
    # 파일의 전체 라인 수 계산
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    if not lines:
        print("❌ 데이터 파일이 비어있습니다")
        return None
    
    # 랜덤 라인 선택
    random_line = random.choice(lines)
    restaurant_data = json.loads(random_line)
    
    return restaurant_data


def test_feature_extraction(model, tokenizer):
    """특징 추출 테스트"""
    
    # 랜덤 레스토랑 데이터 로드
    restaurant_data = load_random_restaurant_data()
    if not restaurant_data:
        return None
    
    description = restaurant_data.get("description", "")
    reviews = restaurant_data.get("reviews", [])
    
    # 리뷰를 \n으로 이어붙이기
    reviews_text = "\n".join(reviews[:30])
    
    print("=== 선택된 레스토랑 데이터 ===")
    print(f"레스토랑명: {restaurant_data.get('title', 'N/A')}")
    print(f"설명: {description[:100]}...")
    print(f"리뷰 개수: {len(reviews)}개")
    print(f"리뷰 텍스트 길이: {len(reviews_text)} 문자")
    print("\n" + "="*50)
    
    test_prompt = f"""식당 소개글과 사용자 리뷰에서 아래 각 항목에 해당하는 특징 키워드가 있으면 추출해주세요.
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
{reviews_text}
"""
    
    print("=== 특징 추출 테스트 ===")
    print(f"입력 프롬프트 길이: {len(test_prompt)} 문자")
    print("\n" + "="*50)
    
    response = generate_response(model, tokenizer, test_prompt)
    
    print("모델 응답:")
    print(response)
    print("\n" + "="*50)
    
    # JSON 파싱 시도
    try:
        result_json = json.loads(response)
        print("JSON 파싱 성공!")
        print(json.dumps(result_json, ensure_ascii=False, indent=2))
        return result_json
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 실패: {e}")
        print("원본 응답을 다시 확인해보세요.")
        return None


def main():
    """메인 실행 함수"""
    
    adapter_path = "yainage90/restaurant-feature-extractor"
    print(adapter_path)
    
    # Hugging Face Hub에서 직접 불러오므로 경로 존재 확인 생략
    
    print("=== 파인튜닝된 Gemma 3 모델 추론 테스트 ===")
    
    # 1. 모델 로드
    print("\n1. 모델 로드 중...")
    try:
        model, tokenizer = load_fine_tuned_model(adapter_path)
        print("✅ 모델 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return
    
    # 2. 특징 추출 테스트
    print("\n2. 특징 추출 테스트 시작...")
    try:
        result = test_feature_extraction(model, tokenizer)
        if result:
            print("✅ 테스트 완료 - JSON 파싱 성공")
        else:
            print("⚠️ 테스트 완료 - JSON 파싱 실패")
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
    
    print("\n=== 추론 테스트 완료 ===")


if __name__ == "__main__":
    main()