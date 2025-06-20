from ingestion.db_pool import QdrantClientPool

def test_qdrant_client_pool():
    pool = QdrantClientPool(path="D:\9D Tech Work\Central_repo\Demo\storage")

    # Acquire client multiple times (they're all the same)
    client1 = pool.acquire()
    client2 = pool.acquire()

    assert client1 is client2, "Expected the same shared client instance"

    pool.release(client1)
    pool.release(client2)

    print("QdrantClientPool test passed.")
