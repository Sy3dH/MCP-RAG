import os
import shutil
from embeddings.vectorizer import FastEmbedder
from ingestion.db_pool import QdrantClientPool
from ingestion.vector_store  import VectorStore

def setup_storage_dir():
    path = "./test_storage"
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path

def test_vector_store_end_to_end():
    storage_path = setup_storage_dir()
    pool = QdrantClientPool(path=storage_path)
    embedder = FastEmbedder()

    docs = [
        {
            "text": "Qdrant is a vector search engine for AI applications.",
            "author": "Alice",
            "findings": "Best for high performance vector search.",
        },
        {
            "text": "FastEmbed provides quick embeddings for your text.",
            "author": "Bob",
            "findings": "Fast and accurate embeddings.",
        },
    ]

    vectors = embedder.embed([d["text"] for d in docs])
    store = VectorStore(pool, vector_size=len(vectors[0]))

    store.init_collection()
    store.ingest(docs, vectors)

    # Test basic query
    query_vector = vectors[0]
    results = store.query(query_vector, limit=2)
    assert len(results) > 0, "Should return at least one result"
    for res in results:
        assert "text" in res.payload
        assert "user_id" in res.payload
        assert "document_id" in res.payload
        assert "chunk_id" in res.payload
        assert "author" in res.payload
        assert "findings" in res.payload

    filtered_results = store.query(query_vector, limit=2, keyword="FastEmbed")
    assert all("FastEmbed" in r.payload["text"] for r in filtered_results), "Filtered results should match keyword"

    print("VectorStore end-to-end test passed.")
