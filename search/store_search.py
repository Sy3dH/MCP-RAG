from typing import List, Optional
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client import QdrantClient
from configs.constants import vector_store_path
from configs.constants import SIMILARITY_THRESHOLD
from qdrant_client.http.exceptions import UnexpectedResponse

def vector_search_with_filter(query_vector: List[float], collection_name: str = "AI_store", limit: int = 5,
                              keyword: Optional[str] = None):
    client = QdrantClient(path=vector_store_path)
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


def is_vector_similar(query_vector: List[float], collection_name: str = "AI_store") -> bool:
    client = QdrantClient(path=vector_store_path)

    try:
        client.get_collection(collection_name=collection_name)
    except ValueError:
        return False

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        score_threshold=SIMILARITY_THRESHOLD,
        limit=1,
        with_payload=False
    )
    return bool(results.points)


def list_vector_stores() -> list:
    client = QdrantClient(path=vector_store_path)
    collections_info = client.get_collections()
    return [collection.name for collection in collections_info.collections]
