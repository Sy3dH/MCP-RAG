from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from typing import List, Optional, Tuple
import uuid
from datetime import datetime
from configs.constants import SIMILARITY_THRESHOLD

class VectorStore:
    def __init__(self, pool, collection_name: str = "AI_store", vector_size: int = 384):
        self.pool = pool
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.author_user_map = {}

    def init_collection(self):
        client = self.pool.acquire()
        try:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        finally:
            self.pool.release(client)

    def _get_or_create_user_id(self, author: str) -> str:
        if author not in self.author_user_map:
            self.author_user_map[author] = f"user_{uuid.uuid4().hex[:8]}"
        return self.author_user_map[author]

    def ingest(self, documents: List[dict], vectors: List[List[float]]) -> Tuple[List[dict], List[dict]]:
        inserted_docs = []
        skipped_docs = []
        points = []
        doc_uuid = f"doc_{uuid.uuid4().hex[:8]}"

        with self.pool.acquire() as client:
            for i, (doc, vector) in enumerate(zip(documents, vectors)):
                search_hits = client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=1,
                    score_threshold=SIMILARITY_THRESHOLD,
                    with_payload=False
                )

                if search_hits:
                    skipped_docs.append(doc)
                    continue

                user_id = self._get_or_create_user_id(doc["author"])
                payload = {
                    "text": doc["text"],
                    "author": doc["author"],
                    "user_id": user_id,
                    "document_id": doc_uuid,
                    "chunk_id": f"chunk_{i}",
                    "time": doc.get("time") or datetime.now(),
                    "findings": doc.get("findings", ""),
                }

                point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
                points.append(point)
                inserted_docs.append(doc)

            if points:
                client.upsert(collection_name=self.collection_name, points=points)

        return inserted_docs, skipped_docs

    def query(self, query_vector: List[float], limit: int = 5, keyword: Optional[str] = None):
        client = self.pool.acquire()
        try:
            filters = None
            if keyword:
                filters = Filter(
                    must=[
                        FieldCondition(key="text", match=MatchValue(value=keyword))
                    ]
                )
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=filters,
            )
            return results
        finally:
            self.pool.release(client)
