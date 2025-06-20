from qdrant_client import QdrantClient
import threading


class QdrantClientPool:
    """
    Singleton wrapper around QdrantClient in local (embedded) mode.
    Ensures only one client accesses the local DB path.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, path="D:\\9D Tech Work\\Central_repo\\Demo\\vector_store"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(QdrantClientPool, cls).__new__(cls)
                cls._instance._init(path)
            return cls._instance

    def _init(self, path):
        self.client = QdrantClient(path=path)

    def acquire(self):
        return self.client

    def release(self, client):
        pass
