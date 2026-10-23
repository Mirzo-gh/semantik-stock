"""Mot de départ à 30 °C (indice optionnel, champ `start` du fichier du jour), DATA_CONTRACT §4b.

Un mot courant, proche de 30 °C pour le mot du jour, qui épargne au joueur la phase froide. Un seul mot par jour pour tous.
Choisi une fois, à la génération du jour, sur les scores STOCKÉS (donc après mélange, filtre et exclusions) : le client ne calcule rien.
"""
import hashlib
import hmac
import pathlib
import unicodedata

import numpy as np

CENTER_STORED = 6000  # 30,00 °C : u = round((°C + 30) × 100)
BAND_HALF_WIDTHS = (100, 200, 400)  # demi-largeur en centièmes de °C : 29-31, puis 28-32, puis 26-34 °C ; jamais plus large
MIN_CANDIDATES = 20  # en dessous, on passe à la bande suivante
MAX_FREQUENCY_RANK = 15_000  # rang de fréquence frWac (pos.tsv) : le joueur doit reconnaître le mot
MIN_LENGTH = 3
COMMON_PREFIX = 4  # lettres de début communes avec le mot du jour = même famille
MAX_EDIT_DISTANCE = 2


def synthetic_lemmas(count: int) -> list[str]:
    """Lemmes factices de 4 lettres (base 26), sans préfixe commun entre eux : pour les vecteurs de test."""
    return ["".join(chr(97 + (index // 26 ** place) % 26) for place in range(4)) for index in range(count)]


def fold(word: str) -> str:
    """Minuscule sans accent (comparaison de familles de mots)."""
    decomposed = unicodedata.normalize("NFD", word.lower().replace("œ", "oe").replace("æ", "ae"))
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def edit_distance(left: str, right: str, limit: int) -> int:
    """Distance de Levenshtein, interrompue dès qu'elle dépasse `limit` (renvoie alors limit + 1)."""
    if abs(len(left) - len(right)) > limit:
        return limit + 1
    previous = list(range(len(right) + 1))
    for row, left_character in enumerate(left, 1):
        current = [row]
        for column, right_character in enumerate(right, 1):
            current.append(min(previous[column] + 1, current[column - 1] + 1, previous[column - 1] + (left_character != right_character)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def is_derivative(candidate: str, target: str) -> bool:
    """Même famille que le mot du jour : début commun de 4 lettres, inclusion (4 lettres ou plus) ou orthographe presque identique."""
    left, right = fold(candidate), fold(target)
    if left[:COMMON_PREFIX] == right[:COMMON_PREFIX] and min(len(left), len(right)) >= COMMON_PREFIX:
        return True
    shorter, longer = sorted((left, right), key=len)
    if len(shorter) >= COMMON_PREFIX and shorter in longer:
        return True
    return edit_distance(left, right, MAX_EDIT_DISTANCE) <= MAX_EDIT_DISTANCE


def read_excluded_words(paths: list[str]) -> set[str]:
    """Mots à ne jamais proposer : première colonne d'un .tsv, ou mots séparés par des espaces d'un .txt ; lignes `#` ignorées."""
    excluded: set[str] = set()
    for path in paths:
        for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            if "\t" in line:
                excluded.add(line.split("\t", 1)[0].strip())
            else:
                excluded.update(line.split())
    return excluded


def pick_order(master_key: bytes, date: str, lemma: str) -> bytes:
    """Rang secret d'un lemme pour ce jour : le plus petit gagne. Déterministe, rien à stocker ; un lemme ajouté ou retiré ne déplace pas les autres."""
    return hmac.new(master_key, f"start30:{date}:{lemma}".encode(), hashlib.sha256).digest()


def candidates_in_band(scores: np.ndarray, half_width: int) -> np.ndarray:
    return np.flatnonzero((scores >= CENTER_STORED - half_width) & (scores <= CENTER_STORED + half_width))


def choose_start(master_key: bytes, date: str, lemmas: list[str], scores: np.ndarray, top: np.ndarray, target_index: int,
                 frequency_ranks: np.ndarray, excluded: set[str]) -> tuple[int | None, int, int]:
    """(index du mot de départ ou None, largeur de bande retenue en centièmes de °C, nombre de candidats de cette bande).

    Candidat : dans la bande, hors du top 1000, rang de fréquence <= MAX_FREQUENCY_RANK, au moins 3 lettres, ni neutralisé (« ~ »),
    ni exclu (noms propres, liste noire, filtre), ni de la famille du mot du jour. Aucun candidat même à la bande la plus large : None.
    """
    in_top = set(int(index) for index in top)
    target = lemmas[target_index]
    count = 0
    for half_width in BAND_HALF_WIDTHS:
        eligible = []
        for index in candidates_in_band(scores, half_width):
            lemma = lemmas[int(index)]
            if (int(index) == target_index or int(index) in in_top or lemma.startswith("~") or len(lemma) < MIN_LENGTH
                    or frequency_ranks[index] > MAX_FREQUENCY_RANK or lemma in excluded or is_derivative(lemma, target)):
                continue
            eligible.append(int(index))
        count = len(eligible)
        if count >= MIN_CANDIDATES or (half_width == BAND_HALF_WIDTHS[-1] and count > 0):
            return min(eligible, key=lambda index: pick_order(master_key, date, lemmas[index])), half_width, count
    return None, BAND_HALF_WIDTHS[-1], count
