"""Score °C et top-1000 d'un puzzle, DATA_CONTRACT §4."""
import numpy as np

TOP_SIZE = 1000
FLOOR_CELSIUS = -30.0
TARGET_CELSIUS = 100.0
TOP_EDGE_CELSIUS = 35.0  # température du lemme le moins proche du top (rang affiché 1)
TARGET_STORED = 13000  # (100 + 30) * 100
MAX_NON_TARGET_STORED = TARGET_STORED - 1  # seul le mot du jour atteint 100 °C


def stored_value(celsius: np.ndarray) -> np.ndarray:
    return np.rint((celsius - FLOOR_CELSIUS) * 100).astype(np.uint16)


def cosine_similarities(vectors: np.ndarray, target_index: int) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    unit = vectors / norms[:, None]
    return unit @ unit[target_index]


def top_indices(similarities: np.ndarray, target_index: int) -> np.ndarray:
    """1000 plus proches, du moins proche au plus proche ; le dernier est le mot du jour.

    Ordre déterministe : similarité puis index (départage les égalités).
    """
    others = np.array([index for index in range(len(similarities)) if index != target_index])
    order = np.lexsort((others, -similarities[others]))  # plus proche d'abord
    nearest = others[order][: TOP_SIZE - 1]
    return np.concatenate([nearest[::-1], [target_index]]).astype(np.uint16)


def celsius_scale(similarities: np.ndarray, top: np.ndarray, target_index: int) -> np.ndarray:
    """Température monotone en similarité, ancrée sur les quantiles du jour.

    médiane -> 0 °C, top[0] -> 35 °C, cible (similarité 1) -> 100 °C, plancher -30 °C.
    """
    median_similarity = float(np.median(similarities))
    edge_similarity = float(similarities[top[0]])
    minimum_similarity = float(similarities.min())
    celsius = np.empty_like(similarities, dtype=np.float64)
    below_median = similarities <= median_similarity
    span_below = max(median_similarity - minimum_similarity, 1e-9)
    celsius[below_median] = FLOOR_CELSIUS * (median_similarity - similarities[below_median]) / span_below
    between = (~below_median) & (similarities < edge_similarity)
    span_between = max(edge_similarity - median_similarity, 1e-9)
    celsius[between] = TOP_EDGE_CELSIUS * (similarities[between] - median_similarity) / span_between
    above = similarities >= edge_similarity
    span_above = max(1.0 - edge_similarity, 1e-9)
    celsius[above] = TOP_EDGE_CELSIUS + (TARGET_CELSIUS - TOP_EDGE_CELSIUS) * (similarities[above] - edge_similarity) / span_above
    return np.clip(celsius, FLOOR_CELSIUS, TARGET_CELSIUS)


def stored_scores(vectors: np.ndarray, target_index: int) -> tuple[np.ndarray, np.ndarray]:
    """(scores uint16 par lemme, top uint16)."""
    return stored_scores_from_similarities(cosine_similarities(vectors, target_index), target_index)


def stored_scores_from_similarities(similarities: np.ndarray, target_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Idem à partir d'une similarité déjà calculée (le mot du jour doit valoir 1, le maximum)."""
    top = top_indices(similarities, target_index)
    stored = stored_value(celsius_scale(similarities, top, target_index))
    stored = np.minimum(stored, MAX_NON_TARGET_STORED)
    stored[target_index] = TARGET_STORED
    return stored, top
