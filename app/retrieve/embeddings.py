from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel
from enum import Enum
import numpy as np


load_dotenv()

client = genai.Client()

EMBEDDING_SIZE = 768


class EmbedTaskType(Enum):
    RETRIEVAL_QUERY: str = "RETRIEVAL_QUERY"
    RETRIEVAL_DOCUMENT: str = "RETRIEVAL_DOCUMENT"


def get_query_embedding(texts: list[str]) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=genai.types.EmbedContentConfig(
            task_type=EmbedTaskType.RETRIEVAL_QUERY,
            output_dimensionality=EMBEDDING_SIZE,
        ),
    )

    embedding = np.array(result.embeddings[0].values)
    embedding = np.array(embedding)
    normed_embedding = embedding / np.linalg.norm(embedding)

    return normed_embedding.tolist()


def get_document_embeddings(texts: list[str]) -> list[list[float]]:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
        config=genai.types.EmbedContentConfig(
            task_type=EmbedTaskType.RETRIEVAL_DOCUMENT,
            output_dimensionality=EMBEDDING_SIZE,
        ),
    )

    embeddings = np.array([emb.values for emb in result.embeddings])
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    return norm_embeddings.tolist()
