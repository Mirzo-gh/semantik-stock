"""Vocabulaire (lemmas.<hash>.txt) et table de formes (forms.<hash>.txt), DATA_CONTRACT §2-3."""
import csv
import hashlib
import re
import unicodedata

PLAYABLE_CATEGORIES = {"NOM", "VER", "ADJ", "ADV"}
MAX_LEMMAS = 65535


def normalize(word: str) -> str:
    return unicodedata.normalize("NFC", word.strip().lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def read_lexique(path: str) -> list[dict]:
    """Lexique 3.83 (TSV) : colonnes ortho, lemme, cgram, freqlivres, freqfilms2."""
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_vocab(rows: list[dict], model_words: set[str], size: int,
                blocklist: set[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """Garde les `size` lemmes les plus fréquents présents dans le modèle.

    Renvoie (lemmes triés par code Unicode, couples (forme, lemme) différents du lemme).
    """
    frequency: dict[str, float] = {}
    for row in rows:
        lemma = normalize(row["lemme"])
        if row["cgram"] not in PLAYABLE_CATEGORIES or " " in lemma or lemma in blocklist:
            continue
        if lemma not in model_words:
            continue
        score = float(row["freqlivres"] or 0) + float(row["freqfilms2"] or 0)
        frequency[lemma] = frequency.get(lemma, 0.0) + score
    kept = sorted(frequency, key=lambda lemma: (-frequency[lemma], lemma))[: min(size, MAX_LEMMAS)]
    lemmas = sorted(kept)
    kept_set = set(kept)
    pairs: dict[str, str] = {}
    for row in rows:
        form, lemma = normalize(row["ortho"]), normalize(row["lemme"])
        if lemma in kept_set and form != lemma and " " not in form and form not in kept_set:
            pairs.setdefault(form, lemma)  # ambiguïté : première entrée gagne
    return lemmas, sorted(pairs.items())


def render_lemmas(lemmas: list[str]) -> str:
    return "\n".join(lemmas) + "\n"


def render_forms(pairs: list[tuple[str, str]], lemmas: list[str]) -> str:
    index = {lemma: position for position, lemma in enumerate(lemmas)}
    return "".join(f"{form};{index[lemma]}\n" for form, lemma in pairs)


TOMBSTONE = "~"  # prefixe d'un lemme neutralise : l'index reste (anciens daily), mais plus aucune saisie ne peut le trouver


SEPARATORS = re.compile(r"[\s\-\u2010-\u2015\u2212_]")  # espace, tiret, traits d'union et tirets typographiques (DATA_CONTRACT §6.1)


def fold(word: str) -> str:
    """Même clé que le client (DATA_CONTRACT §6.1) : NFC, minuscule, œ/æ, sans diacritiques, puis sans tiret ni espace (l'apostrophe reste)."""
    word = unicodedata.normalize("NFC", word.strip().lower()).replace("œ", "oe").replace("æ", "ae").replace("’", "'")
    folded = "".join(char for char in unicodedata.normalize("NFD", word) if not unicodedata.category(char).startswith("M"))
    return SEPARATORS.sub("", folded)


def separator_collisions(entries: list[str]) -> dict[str, list[str]]:
    """Clés partagées par deux entrées distinctes dont l'une porte un tiret ou une espace (« week-end » / « weekend »).

    Les doublons d'accent (« ete » / « été ») ne sont pas ici : find_junk les traite. Le build s'arrête si ce dictionnaire n'est pas vide.
    """
    groups: dict[str, set[str]] = {}
    for entry in entries:
        groups.setdefault(fold(entry), set()).add(entry)
    return {key: sorted(group) for key, group in groups.items() if len(group) > 1 and any(SEPARATORS.search(entry) for entry in group)}


def find_junk(rows: list[tuple], form_folds: set[str], protected: set[str]) -> set[str]:
    """Lemmes mal accentués ou doublons d'accent (« ete », « eté » à côté de « été »).

    rows : (lemme, pos, rang de fréquence, vecteur, connu du dictionnaire). Dans un groupe de lemmes qui se replient sur la
    même chaîne, on neutralise ceux qui sont inconnus du dictionnaire et soit moins fréquents que le meilleur du groupe,
    soit couverts par une forme fléchie du même repli (« été » → « être »). Jamais un mot protégé (candidats du jour).
    """
    groups: dict[str, list[tuple]] = {}
    for row in rows:
        groups.setdefault(fold(row[0]), []).append(row)
    junk = set()
    for key, members in groups.items():
        if len(members) < 2 and key not in form_folds:
            continue
        best = min(members, key=lambda row: row[2])
        for row in members:
            if row[0] in protected or row[4]:
                continue
            if row is not best or key in form_folds:
                junk.add(row[0])
    return junk
