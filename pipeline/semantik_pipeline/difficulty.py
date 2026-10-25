"""Difficulté d'un mot du jour : nombre d'essais d'un joueur simulé (étude hive/work/semantik/algo/PROPRE-DIFFICULTE.md).

- Le joueur joue d'abord 40 sondes courantes (médoïdes d'un k-means sur les 3 000 lemmes les plus fréquents), puis à chaque essai le lemme courant
  non encore joué le plus proche (cosinus frWac) de la moyenne de ses 3 meilleurs essais, au vu du score réel du jour. Difficulté = nombre d'essais
  jusqu'au mot, plafonné à MAX_GUESSES + 1 (non trouvé). Médiane de trois jeux de sondes (la simulation est bruitée : écart-type ~34 essais).
- Classes : f (facile) < 75 essais, m (moyen) 75 à 185, d (difficile) > 185 et jusqu'à MAX_GUESSES, x (non trouvé en MAX_GUESSES essais).
- pipeline/difficulty.tsv : `mot<TAB>essais<TAB>classe`, calculé hors CI (make_difficulty.py) ; ne dit rien de l'ordre du calendrier.
"""
import pathlib

import numpy as np

MAX_GUESSES = 400
PROBE_COUNT = 40
PROBE_SEEDS = (1, 2, 3)
POOL_SIZE = 20_000
EASY_BELOW = 75
HARD_ABOVE = 185


def difficulty_class(guesses: float) -> str:
    if guesses > MAX_GUESSES:
        return "x"
    if guesses > HARD_ABOVE:
        return "d"
    return "f" if guesses < EASY_BELOW else "m"


def make_probes(unit_vectors: np.ndarray, pool_indexes: np.ndarray, seed: int) -> list[int]:
    top = pool_indexes[:3000]
    matrix = unit_vectors[top]
    rng = np.random.default_rng(seed)
    centroids = matrix[rng.choice(len(matrix), PROBE_COUNT, replace=False)].copy()
    for _ in range(20):
        labels = (matrix @ centroids.T).argmax(1)
        for cluster in range(PROBE_COUNT):
            if (labels == cluster).any():
                centroids[cluster] = matrix[labels == cluster].mean(0)
                centroids[cluster] /= np.linalg.norm(centroids[cluster])
    return [int(top[(matrix @ centroids[cluster]).argmax()]) for cluster in range(PROBE_COUNT)]


def play(scores: np.ndarray, target: int, unit_vectors: np.ndarray, pool_indexes: np.ndarray, probes: list[int], best_count: int = 3) -> int:
    """Nombre d'essais du joueur simulé (MAX_GUESSES + 1 s'il ne trouve pas) ; `scores` : plus haut = plus chaud."""
    pool_matrix = unit_vectors[pool_indexes]
    position = {int(index): rank for rank, index in enumerate(pool_indexes)}
    alive = np.ones(len(pool_indexes), dtype=bool)
    guessed: list[int] = []

    def mark(index: int) -> None:
        guessed.append(index)
        if index in position:
            alive[position[index]] = False

    for probe in probes:
        if probe == target:
            return len(guessed) + 1
        mark(probe)
    while len(guessed) < MAX_GUESSES:
        best = np.array(guessed)[np.argsort(-scores[guessed])[:best_count]]
        center = unit_vectors[best].mean(0)
        center /= np.linalg.norm(center)
        similarities = pool_matrix @ center
        similarities[~alive] = -9
        choice = int(pool_indexes[int(similarities.argmax())])
        mark(choice)
        if choice == target:
            return len(guessed)
    return MAX_GUESSES + 1


def median_guesses(scores: np.ndarray, target: int, unit_vectors: np.ndarray, pool_indexes: np.ndarray, probe_sets: list[list[int]]) -> int:
    return int(np.median([play(scores, target, unit_vectors, pool_indexes, probes) for probes in probe_sets]))


def parse_difficulty(text: str) -> dict[str, tuple[int, str]]:
    """pipeline/difficulty.tsv -> {mot: (essais, classe)}."""
    table: dict[str, tuple[int, str]] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        word, guesses, difficulty = line.split("\t")
        table[word] = (int(guesses), difficulty)
    return table


DRAW_EXCLUSIONS_NAME = "draw_exclusions.txt"
DRAW_KEEP_NAME = "candidates_keep.txt"
DRAW_FROM_NAME = "draw_exclusions_from.txt"
DRAW_ADDITIONS_NAME = "draw_additions.txt"
DRAW_ADDITIONS_FROM_NAME = "draw_additions_from.txt"


def _read_word_file(path: pathlib.Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))


def read_draw_exclusions(difficulty_path) -> frozenset[str]:
    """Mots retirés du TIRAGE (trop peu courants, décision humaine) : fichier `draw_exclusions.txt` à côté de difficulty.tsv, un mot par ligne (# = commentaire),
    moins les mots rétablis à la main (`candidates_keep.txt`, même dossier). Absent : rien n'est retiré. Jamais appliqué aux jours déjà fabriqués."""
    folder = pathlib.Path(difficulty_path).parent
    return _read_word_file(folder / DRAW_EXCLUSIONS_NAME) - _read_word_file(folder / DRAW_KEEP_NAME)


def read_draw_exclusions_from(difficulty_path) -> int:
    """Premier jour (numéro) où les exclusions du tirage s'appliquent (`draw_exclusions_from.txt`) ; absent : dès le jour 1."""
    path = pathlib.Path(difficulty_path).with_name(DRAW_FROM_NAME)
    return int(path.read_text(encoding="utf-8").split()[0]) if path.exists() else 1


def read_draw_additions(difficulty_path) -> tuple[frozenset[str], int]:
    """Candidats AJOUTÉS au tirage (décision humaine du 25/10) : `draw_additions.txt` (un mot par ligne) et `draw_additions_from.txt` (premier jour qui peut les tirer),
    à côté de difficulty.tsv. Les mots figurent aussi dans candidates.tsv / difficulty.tsv ; avant ce jour ils sont retirés du tirage, pour que les jours déjà fabriqués ne bougent pas."""
    folder = pathlib.Path(difficulty_path).parent
    words = _read_word_file(folder / DRAW_ADDITIONS_NAME)
    path = folder / DRAW_ADDITIONS_FROM_NAME
    return words, (int(path.read_text(encoding="utf-8").split()[0]) if words and path.exists() else 1)


def keep_candidates(candidates: list[str], table: dict[str, tuple[int, str]], dropped_classes: str, excluded: frozenset[str] = frozenset()) -> list[str]:
    """Candidats sans les classes retirées (ex. « dx ») ni les mots exclus du tirage. Un candidat absent de la table reste : on ne retire jamais en silence."""
    return [word for word in candidates if table.get(word, (0, "m"))[1] not in dropped_classes and word not in excluded]
