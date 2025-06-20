from typing import List, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from configs.constants import vector_store_path
from embeddings.vectorizer import FastEmbedder
from configs.constants import SIMILARITY_THRESHOLD
from qdrant_client.http.exceptions import UnexpectedResponse

def vector_search_with_filter(query_vector: List[float], collection_name: str = "AI_store", limit: int = 5, keyword: Optional[str] = None):
    client = QdrantClient(path = vector_store_path)
    try:
        filters = None
        if keyword:
            filters = Filter(
                must=[
                    FieldCondition(key="text", match=MatchValue(value=keyword))
                ]
            )
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            query_filter=filters
        )
        return results

    except UnexpectedResponse as e:
       raise RuntimeError(f"Qdrant query failed: {str(e)}") from e
    except Exception as e:
       raise RuntimeError("An unexpected error occurred during vector search.") from e


def vector_search_with_threshold(query_vector: List[float], collection_name: str = "AI_store"):
        client = QdrantClient(path=vector_store_path)

        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            score_threshold=SIMILARITY_THRESHOLD,
        )
        return results
