"""Lecteur minimal du format word2vec binaire (frWac), sans gensim."""
import numpy as np


def load_word2vec_binary(path: str, wanted: set[str] | None = None) -> dict[str, np.ndarray]:
    """Renvoie {mot: vecteur float32}. `wanted` filtre pour limiter la mémoire."""
    vectors: dict[str, np.ndarray] = {}
    with open(path, "rb") as handle:
        count, dimension = (int(value) for value in handle.readline().split())
        for _ in range(count):
            word_bytes = bytearray()
            while True:
                char = handle.read(1)
                if char in (b" ", b""):
                    break
                if char != b"\n":
                    word_bytes += char
            raw = handle.read(4 * dimension)
            word = word_bytes.decode("utf-8", errors="ignore")
            if wanted is None or word in wanted:
                vectors[word] = np.frombuffer(raw, dtype="<f4").copy()
    return vectors
