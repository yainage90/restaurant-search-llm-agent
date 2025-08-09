"""
맛집 검색 LLM 에이전트 유틸리티 함수들
"""

import json
import re
from functools import wraps
from typing import Any, Callable
from google import genai
from .config import ui_messages


# Gemini 클라이언트 초기화
client = genai.Client()


def handle_exceptions(
    default_return: Any = None,
    error_prefix: str = ui_messages.error_prefix,
    log_error: bool = True
):
    """
    예외 처리 데코레이터
    
    Args:
        default_return: 예외 발생 시 반환할 기본값
        error_prefix: 에러 메시지 접두사
        log_error: 에러 로그 출력 여부
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    print(f"[ERROR] {func.__name__}: {str(e)}")
                
                if isinstance(default_return, str):
                    return f"{error_prefix}{str(e)}"
                elif isinstance(default_return, dict):
                    return {**default_return, "error": str(e)}
                else:
                    return default_return
        return wrapper
    return decorator


def parse_json_response(response: str) -> dict:
    """
    LLM 응답에서 JSON을 안전하게 파싱
    
    Args:
        response: LLM 응답 텍스트
        
    Returns:
        dict: 파싱된 JSON 또는 에러 정보
    """
    try:
        # JSON 블록 찾기
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"error": "JSON 블록을 찾을 수 없습니다"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON 파싱 오류: {str(e)}"}
    except Exception as e:
        return {"error": f"예상치 못한 오류: {str(e)}"}


def safe_format_json(data: Any, ensure_ascii: bool = False, indent: int = 2) -> str:
    """
    안전한 JSON 포맷팅
    
    Args:
        data: JSON으로 변환할 데이터
        ensure_ascii: ASCII 문자만 사용할지 여부
        indent: 들여쓰기 크기
        
    Returns:
        str: 포맷팅된 JSON 문자열
    """
    try:
        return json.dumps(data, ensure_ascii=ensure_ascii, indent=indent)
    except Exception as e:
        return f"{ui_messages.error_prefix}JSON 변환 실패: {str(e)}"


def calculate_keyword_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트 간의 키워드 유사도 계산
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
        
    Returns:
        float: 0.0-1.0 사이의 유사도 값
    """
    if not text1 or not text2:
        return 0.0
    
    keywords1 = set(text1.lower().split())
    keywords2 = set(text2.lower().split())
    
    intersection = len(keywords1 & keywords2)
    union = len(keywords1 | keywords2)
    
    return intersection / union if union > 0 else 0.0


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    텍스트를 지정된 길이로 자르기
    
    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        suffix: 잘린 부분을 표시할 접미사
        
    Returns:
        str: 잘린 텍스트
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def format_timestamp(dt, format_str: str = "%H:%M:%S") -> str:
    """
    날짜시간을 지정된 형식으로 포맷팅
    
    Args:
        dt: datetime 객체
        format_str: 포맷 문자열
        
    Returns:
        str: 포맷팅된 시간 문자열
    """
    try:
        return dt.strftime(format_str)
    except Exception:
        return "시간 정보 없음"


def create_session_id(identifier: str) -> str:
    """
    식별자를 기반으로 안전한 세션 ID 생성
    
    Args:
        identifier: 기본 식별자 (IP 주소 등)
        
    Returns:
        str: 생성된 세션 ID
    """
    return f"user_{hash(str(identifier))}"


def generate_decision(prompt: str) -> str:
    """
    검색 필요성 판단 등 간단한 의사결정을 위한 경량화된 생성 함수
    
    Args:
        prompt: 판단을 위한 프롬프트
        
    Returns:
        LLM이 생성한 응답 텍스트
        
    Raises:
        Exception: LLM 호출 중 오류가 발생한 경우
    """
    
    # 간단한 의사결정용 시스템 프롬프트
    decision_system_prompt = """당신은 사용자의 요청을 분석하여 명확하고 정확한 판단을 내리는 AI입니다.
주어진 지침에 따라 정확한 JSON 형태로 응답해주세요."""
    
    # LLM에 요청
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=decision_system_prompt,
            max_output_tokens=512,
        )
    )
    
    return response.text