from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai

load_dotenv()

client = TavilyClient()

# client = genai.Client()
# Define the grounding tool
# grounding_tool = genai.types.Tool(
#     google_search=genai.types.GoogleSearch(),
# )


def search_with_tavily(query: str):
    response = client.search(
        query=query,
        max_results=3,
        search_depth='basic',
    )

    docs = [
        {
            "title": result["title"],
            "content": result["content"],
        } for result in response["results"]
    ]

    return docs 


# def search_with_gemini(query: str):
#     response = client.models.generate_content(
#         model="gemini-2.5-flash-lite",
#         contents=query,
#         config = genai.types.GenerateContentConfig(
#             temperature=0.0,
#             tools=[grounding_tool],
#         )
#     )

#     return response


def test_queries():
    queries = [
        "김두한이 자주가던 국밥집",
        "무한도전 기사식당",
    ]

    for query in queries:
        response = search_with_tavily(query)
        # response = search_with_gemini(query)
        print(response)
        print('-' * 80)


if __name__ == "__main__":
    test_queries()
    