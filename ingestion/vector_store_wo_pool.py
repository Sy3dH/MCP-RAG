from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from models.models import DocumentModel
from datetime import datetime
from configs.constants import SIMILARITY_THRESHOLD
import os
from dotenv import load_dotenv
import uuid

load_dotenv()
class VectorStore:
    def __init__(self, collection_name: str = "AI_store", vector_size: int = 384):
        self.client = QdrantClient(path="D:\\9D Tech Work\\Central_repo\\Demo\\vector_store")
    # QdrantClient(url="https://43d5eb52-cd13-4a0b-8944-26d667ca16da.europe-west3-0.gcp.cloud.qdrant.io:6333",
    # api_key=os.getenv("QDRANT_API_KEY"))
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.author_user_map = {}

    def init_collection(self):
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )

    def _get_or_create_user_id(self, author: str) -> str:
        if author not in self.author_user_map:
            self.author_user_map[author] = f"user_{uuid.uuid4().hex[:8]}"
        return self.author_user_map[author]

    def ingest(self, documents: List[DocumentModel], vectors: List[List[float]]):
        inserted_docs = []
        skipped_docs = []
        points = []
        doc_uuid = f"doc_{uuid.uuid4().hex[:8]}"

        for i, (doc, vector) in enumerate(zip(documents, vectors)):
            search_hits = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=1,
                score_threshold=SIMILARITY_THRESHOLD,
                with_payload=False
            )

            if search_hits.points:
                skipped_docs.append(doc)
                continue

            user_id = self._get_or_create_user_id(doc.researcher)
            payload = {
                "text": doc.text,
                "researcher": doc.researcher,
                "user_id": user_id,
                "document_id": doc_uuid,
                "chunk_id": f"chunk_{i}",
                "time": datetime.now(),
                "pdf_path": doc.pdf_path,
                "findings": doc.findings
            }

            point = PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)
            points.append(point)
            inserted_docs.append(doc)

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        return inserted_docs, skipped_docs

    # def ingest(self, documents: List[DocumentModel], vectors: List[List[float]]):
    #     doc_uuid = f"doc_{uuid.uuid4().hex[:8]}"
    #     points = []
    #
    #     for i, (doc, vector) in enumerate(zip(documents, vectors)):
    #         user_id = self._get_or_create_user_id(doc.researcher)
    #         payload = {
    #             "text": doc.text,
    #             "researcher": doc.researcher,
    #             "user_id": user_id,
    #             "document_id": doc_uuid,
    #             "chunk_id": f"chunk_{i}",
    #             "time": doc.time,
    #             "findings": doc.findings,
    #         }
    #         points.append(PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload))
    #
    #     self.client.upsert(collection_name=self.collection_name, points=points)