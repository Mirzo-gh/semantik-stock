"""Changer le mot d'un jour futur (tableau de bord admin) : contrôles et choix du mot, sans jamais l'écrire dans un journal.

Fonctions pures, appelées par `cli change` (qui écrit le résultat dans un fichier PRIVÉ) et par `cli daily --force-word-file`.
Les codes de refus sont des mots-clés stables (l'écran les traduit) ; aucun ne contient ni ne dérive du mot.
"""
import json
import pathlib

from . import daily as daily_module

MODES = ("word", "redraw", "revert")
REFUSAL_TEXTS = {
    "not_in_vocabulary": "mot absent du vocabulaire ou neutralisé",
    "not_candidate": "mot non retenu dans la liste des candidats (liste ou classe de difficulté)",
    "recently_played": "mot déjà joué ce cycle ou dans les derniers jours",
    "history_incomplete": "historique des jours précédents incomplet",
    "bad_request": "demande illisible",
}


def history_words(daily_dir: pathlib.Path, lemmas: list[str], before: int) -> list[str] | None:
    """Mots des jours 1..before-1 d'après les fichiers daily ; None si l'un manque."""
    words = []
    for number in range(1, before):
        path = daily_dir / f"{number}.json"
        if not path.exists():
            return None
        words.append(lemmas[daily_module.decode_target_index(json.loads(path.read_text()))])
    return words


def word_problems(word: str, history: list[str], lemma_set: set[str], draw_candidates: list[str]) -> list[str]:
    """Codes de refus d'un mot imposé ; liste vide = jouable."""
    if word not in lemma_set or word.startswith("~"):
        return ["not_in_vocabulary"]
    if word not in set(draw_candidates):
        return ["not_candidate"]
    _, played = daily_module.replay(history)
    if word in played or word in history[-(daily_module.MIN_GAP_DAYS - 1):]:
        return ["recently_played"]
    return []


def redraw_word(master_key: bytes, nonce: str, history: list[str], draw_candidates: list[str], old_word: str) -> str:
    """Autre mot, tiré sur la même chaîne avec une clé dérivée du nonce, l'ancien mot exclu (déterministe : même demande, même mot)."""
    pool = [word for word in draw_candidates if word != old_word]
    return daily_module.next_word(master_key + b"|redraw|" + nonce.encode(), history, pool)


def reviewed_words(exclusions_text: str | None) -> set[str]:
    """Mots dont les voisins ont été relus (première colonne de exclusions.tsv)."""
    if not exclusions_text:
        return set()
    return {line.split("\t")[0] for line in exclusions_text.splitlines() if line.strip() and not line.startswith("#")}


def resolve(request: dict, history: list[str] | None, lemmas: list[str], draw_candidates: list[str], old_word: str, master_key: bytes, exclusions_text: str | None) -> dict:
    """{codes, word, neighbours_unreviewed} ; word est PRIVÉ (fichier de travail, jamais affiché). `revert` ne demande aucun mot."""
    mode = request.get("mode")
    if mode not in MODES:
        return {"codes": ["bad_request"], "word": None, "neighbours_unreviewed": None}
    if mode == "revert":
        return {"codes": [], "word": None, "neighbours_unreviewed": None}
    if history is None:
        return {"codes": ["history_incomplete"], "word": None, "neighbours_unreviewed": None}
    if mode == "word":
        word = request.get("word")
        if not isinstance(word, str) or not word.strip():
            return {"codes": ["bad_request"], "word": None, "neighbours_unreviewed": None}
        word = word.strip()
    else:
        nonce = request.get("nonce")
        if not isinstance(nonce, str) or not nonce:
            return {"codes": ["bad_request"], "word": None, "neighbours_unreviewed": None}
        word = redraw_word(master_key, nonce, history, draw_candidates, old_word)
    codes = word_problems(word, history, set(lemmas), draw_candidates)
    if codes:
        return {"codes": codes, "word": None, "neighbours_unreviewed": None}
    return {"codes": [], "word": word, "neighbours_unreviewed": word not in reviewed_words(exclusions_text)}
