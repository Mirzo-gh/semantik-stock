"""Assemblage du fichier /data/daily/<n>.json, DATA_CONTRACT §4."""
import base64
import hashlib
import hmac
import json
import numpy as np

from . import CONTRACT_VERSION
from .masking import xor_mask
from .scoring import stored_scores, stored_scores_from_similarities


def derive_seed(master_key: bytes, date: str) -> int:
    """Graine du masque : uint32 tiré par HMAC(clé maîtresse, "seed:" + date)."""
    digest = hmac.new(master_key, b"seed:" + date.encode(), hashlib.sha256).digest()
    return int.from_bytes(digest[:4], "big")


MIN_GAP_DAYS = 30  # un même mot ne revient jamais à moins de 30 jours (y compris d'un cycle à l'autre)


def word_score(master_key: bytes, cycle: int, word: str) -> bytes:
    """Rang secret d'un mot dans un cycle : plus petit = tiré plus tôt. Chaque mot a son propre score, donc ajouter un mot ne déplace aucun autre."""
    return hmac.new(master_key, f"cycle:{cycle}:{word}".encode(), hashlib.sha256).digest()


def replay(history: list[str]) -> tuple[int, set[str]]:
    """(numéro de cycle, mots déjà joués dans ce cycle) reconstitués depuis l'historique seul.

    Un nouveau cycle commence exactement quand un mot déjà joué revient (le tirage ne le fait que quand le réservoir est vide).
    """
    cycle, played = 0, set()
    for word in history:
        if word in played:
            cycle, played = cycle + 1, set()
        played.add(word)
    return cycle, played


def next_word(master_key: bytes, history: list[str], candidates: list[str]) -> str:
    """Mot du prochain puzzle : le premier, dans l'ordre secret du cycle, des mots pas encore joués ce cycle.

    - Liste étendue en cours de cycle : les nouveaux mots rejoignent le réservoir restant, rien d'autre ne bouge.
    - Réservoir vide : nouveau cycle sur toute la liste.
    - On évite les mots des 29 derniers jours (jonction de cycles) tant qu'il reste un autre choix.
    """
    cycle, played = replay(history)
    pool = [word for word in sorted(set(candidates)) if word not in played]
    if not pool:
        cycle, pool = cycle + 1, sorted(set(candidates))
    recent = set(history[-(MIN_GAP_DAYS - 1):])
    allowed = [word for word in pool if word not in recent] or pool
    return min(allowed, key=lambda word: (word_score(master_key, cycle, word), word))


def decode_target_index(puzzle: dict) -> int:
    """Index du lemme du jour dans un fichier daily (dernier élément de `top`, après démasquage)."""
    raw = xor_mask(base64.b64decode(puzzle["top"]), puzzle["seed"])
    return int.from_bytes(raw[-2:], "little")


def build_daily(vectors: np.ndarray, target_index: int, number: int, date: str,
                vocab_hash: str, forms_hash: str, seed: int, similarities: np.ndarray | None = None,
                pos_class: int | None = None) -> dict:
    """`similarities` (score mélangé, voir blend.py) remplace le cosinus frWac ; sans lui, le calcul historique est inchangé."""
    if similarities is None:
        scores, top = stored_scores(vectors, target_index)
    else:
        scores, top = stored_scores_from_similarities(similarities, target_index)
    puzzle = {
        "v": CONTRACT_VERSION,
        "n": number,
        "date": date,
        "vocab": vocab_hash,
        "forms": forms_hash,
        "seed": seed,
        "scores": base64.b64encode(xor_mask(scores.astype("<u2").tobytes(), seed)).decode(),
        "top": base64.b64encode(xor_mask(top.astype("<u2").tobytes(), seed)).decode(),
    }
    if pos_class is not None:
        # Indice « nature » (optionnel, DATA_CONTRACT) : un octet masqué avec un flux neuf, comme scores et top
        puzzle["pos"] = base64.b64encode(xor_mask(bytes([pos_class]), seed)).decode()
    return puzzle


def decode_pos(puzzle: dict) -> int | None:
    """Classe de nature du mot du jour (0 nom, 1 verbe, 2 adjectif, 3 adverbe, 4 autre) ; None si le fichier n'en a pas."""
    if "pos" not in puzzle:
        return None
    return xor_mask(base64.b64decode(puzzle["pos"]), puzzle["seed"])[0]


def dump(daily: dict) -> str:
    return json.dumps(daily, separators=(",", ":"))
