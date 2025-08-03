import gradio as gr
from dotenv import load_dotenv
from typing import Iterator
from ..retrieve.search import search
from ..generation.generation import generate, generate_streaming

load_dotenv()

# 대화 상태를 저장할 전역 변수
conversation_context = None

def chat_fn(message: str, history: list[list[str]]) -> str:
    """
    채팅 처리 함수
    
    Args:
        message: 사용자의 현재 메시지
        history: 이전 대화 기록 [[user_msg, bot_response], ...]
        
    Returns:
        str: 응답 메시지
    """
    global conversation_context

    if not history:
        # 첫 대화: 검색을 수행하고 결과를 context에 저장
        query = message
        restaurant_context = search(query)
        conversation_context = restaurant_context  # 검색 결과를 전역 컨텍스트에 저장

        bot_response = generate(query, restaurant_context)
        print(f'bot_response: {bot_response}')
        return bot_response
    else:
        # 이어지는 대화: 저장된 context와 전체 대화 기록을 함께 사용
        # 대화 히스토리를 LLM에 제공할 프롬프트 형식으로 변환
        chat_history_prompt = ""
        for user_msg, bot_msg in history:
            chat_history_prompt += f"사용자: {user_msg}\n"
            chat_history_prompt += f"어시스턴트: {bot_msg}\n"
        
        # 현재 사용자 메시지 추가
        chat_history_prompt += f"사용자: {message}\n"

        # 저장된 레스토랑 정보와 대화 기록을 모두 컨텍스트로 활용
        response_text = generate(
            query="",
            context=f"참고 정보: {conversation_context}\n\n---대화 기록---\n{chat_history_prompt}"
        )
        return response_text


def create_chat_interface():
    """Gradio 채팅 인터페이스 생성"""

    # Gradio 테마 커스터마이징 (한글 폰트 및 사이즈 조정)
    custom_theme = gr.themes.Default(
        font=(gr.themes.GoogleFont("Noto Sans KR"), "ui-sans-serif", "system-ui", "sans-serif"),
        text_size=gr.themes.sizes.text_sm,
        spacing_size=gr.themes.sizes.spacing_sm,
    )
    
    # Gradio ChatInterface 사용
    chat_interface = gr.ChatInterface(
        fn=chat_fn,
        title="🍽️ 식당 추천 AI 어시스턴트",
        description="안녕하세요! 식당을 찾고 설명해드리는 AI 어시스턴트입니다. 원하는 음식, 지역, 분위기 등을 말씀해주세요!",
        examples=[
            "강남역 근처 맛있는 한식당 추천해줘",
            "데이트하기 좋은 이태리 레스토랑이 어디있을까?",
            "홍대에서 혼밥하기 좋은 곳 알려줘",
            "회식하기 좋은 고기집 추천해줘"
        ],
        cache_examples=False,
        theme=custom_theme,
    )
    
    return chat_interface

if __name__ == "__main__":
    interface = create_chat_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )