"""
OpenAI embeddings. Claude has no embeddings endpoint, so this is the one
place OpenAI is used - everything else (routing, generation, groundedness)
runs on Claude.
"""
from openai import OpenAI

from config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts. OpenAI's embeddings endpoint accepts up
    to 2048 inputs per call; batch here defensively in case a page produces
    more chunks than that."""
    if not texts:
        return []

    all_vectors: list[list[float]] = []
    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = _client.embeddings.create(model=settings.embedding_model, input=batch)
        all_vectors.extend([d.embedding for d in resp.data])
    return all_vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
