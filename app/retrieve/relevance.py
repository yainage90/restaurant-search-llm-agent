"""
자연어 질의와 검색된 문서들의 연관도를 LLM(Gemini 2.5 Flash Lite)으로 판단하는 모듈
"""

import json
from typing import Any
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
from app.llm.llm import generate_with_gemini

load_dotenv()


llm = genai.Client()


SYSTEM_PROMPT = """
당신은 사용자의 질의와 검색된 문서들 간의 연관성을 정확하게 판단하는 AI 어시스턴트입니다.
주어진 정보를 바탕으로 각 문서의 관련성을 평가하고, 전체적인 연관성 판단과 그 근거를 명확하게 제시해야 합니다.
"""


def create_relevance_prompt(query: str, documents: list[dict[str, Any]]) -> str:
    """연관도 평가를 위한 프롬프트 생성"""
    
    documents_text = ""
    for i, doc in enumerate(documents, 1):
        doc_info = f"""문서 {i}:
{doc.get('summary', 'N/A')}...\n
"""
        documents_text += doc_info
    
    prompt = f"""다음은 사용자의 질의와 검색된 식당 문서들입니다.

사용자 질의: "{query}"

검색된 문서들:
{documents_text}

각 문서가 사용자의 질의에 얼마나 관련성이 있는지 평가해주세요.

평가 기준:
- "relevant": 문서가 사용자의 질의에 직접적으로 답할 수 있는 정보를 포함
- "irrelevant": 문서가 사용자의 질의와 관련이 없거나 답할 수 없음

응답 형식 (JSON):
{{
    "overall_relevance": "relevant" 또는 "irrelevant",
    "reason": "판단 근거",
    "document_scores": [
        {{
            "document_id": "1",
            "relevance": "relevant" 또는 "irrelevant",
            "reason": "해당 문서의 판단 근거"
        }}
    ]
}}

reason은 한국어로 작성해주세요."""
    
    return prompt


class RelevanceResult(BaseModel):
    overall_relevance: str
    reason: str
    document_scores: list


def grade_relevance(query: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
    """
    자연어 질의와 검색된 문서들의 연관도를 평가
    
    Args:
        query: 사용자의 자연어 질의
        documents: 검색된 문서 리스트
        
    Returns:
        연관도 평가 결과
    """
    
    if not documents:
        return {
            "overall_relevance": "irrelevant",
            "reason": "검색된 문서가 없습니다.",
            "document_scores": []
        }
    
    # 프롬프트 생성
    prompt = create_relevance_prompt(query, documents)
    
    result = generate_with_gemini(
        model="gemini-2.5-flash",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=prompt,
        max_output_tokens=1024,
        response_shcema=RelevanceResult,
    )
    
    # 결과 검증
    if "overall_relevance" not in result:
        raise ValueError("응답에 overall_relevance 필드가 없습니다.")
    
    if result["overall_relevance"] not in ["relevant", "irrelevant"]:
        raise ValueError("overall_relevance 값이 유효하지 않습니다.")
    
    return result
        

def test_relevance_grading():
    """연관도 평가 기능 테스트"""
    
    # 테스트 문서 생성
    test_documents = [
        {
            "title": "스시히로바",
            "address": "서울특별시 강남구 테헤란로 123",
            "menus": [
                {"name": "초밥세트", "price": 35000},
                {"name": "사시미", "price": 45000}
            ],
            "convenience": ["주차", "예약"],
            "atmosphere": ["고급스러운", "조용한"],
            "occasion": ["데이트", "비즈니스"],
            "review_food": ["초밥", "사시미", "와사비"],
            "features": ["신선한 재료", "숙련된 셰프"],
            "summary": "강남 최고의 일식 전문점으로 신선한 초밥과 사시미를 제공합니다."
        },
        {
            "title": "맘스터치 강남점",
            "address": "서울특별시 강남구 역삼동 456",
            "menus": [
                {"name": "싸이버거", "price": 8000},
                {"name": "치킨버거", "price": 7500}
            ],
            "convenience": ["포장", "배달"],
            "atmosphere": ["캐주얼한"],
            "occasion": ["혼밥", "간편식사"],
            "review_food": ["햄버거", "치킨", "감자튀김"],
            "features": ["빠른 서비스", "저렴한 가격"],
            "summary": "강남 지역의 인기 햄버거 체인점으로 합리적인 가격의 버거를 제공합니다."
        }
    ]
    
    test_queries = [
        "강남역 주차되는 일식집",
        "강남 햄버거 맛집",
        "이태원 파스타 맛집"
    ]
    
    print("=== 연관도 평가 테스트 ===")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. 테스트 쿼리: '{query}'")
        print("-" * 50)
        
        result = grade_relevance(query, test_documents)
        
        print(f"전체 연관도: {result['overall_relevance']}")
        print(f"판단 근거: {result['reason']}")
        
        if result.get('document_scores'):
            print("\n개별 문서 점수:")
            for score in result['document_scores']:
                doc_id = score.get('document_id', 'N/A')
                doc_title = test_documents[doc_id - 1]['title'] if isinstance(doc_id, int) and 1 <= doc_id <= len(test_documents) else 'N/A'
                print(f"  문서 {doc_id} ({doc_title}): {score.get('relevance', 'N/A')}")
                print(f"    근거: {score.get('reason', 'N/A')}")
        
        print()


if __name__ == "__main__":
    test_relevance_grading()