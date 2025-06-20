from embeddings.vectorizer import FastEmbedder
import numpy as np

def test_fast_embedder_basic():
    embedder = FastEmbedder()

    documents = [
        "FastEmbed is optimized for speed.",
        "It can be used in production environments."
    ]

    vectors = embedder.embed(documents)

    # Check output size
    assert len(vectors) == len(documents), "Output vector count should match document count"

    # Check vector dimensions
    expected_dim = len(vectors[0])
    assert all(len(vec) == expected_dim for vec in vectors), "All vectors should have same dimension"
    # Check that it's a list of lists of floats

    assert isinstance(vectors[0], np.ndarray), "Each vector should be a list"
    assert isinstance(vectors[0][0], np.float32), "Each vector element should be a float"

    print("FastEmbedder insertion test passed.")

def test_fast_embedder_empty_input():
    embedder = FastEmbedder()
    vectors = embedder.embed([])
    assert vectors == [], "Empty input should return empty output"
    print("FastEmbedder empty vector test passed.")
