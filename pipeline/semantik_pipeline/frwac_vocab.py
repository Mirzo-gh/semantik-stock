"""Vocabulaire depuis frWac lemmatisé (jetons `lemme_pos`, triés par fréquence) : aucune dépendance à Lexique.

Les formes fléchies viennent du vocabulaire du modèle non lemmatisé, ramenées au lemme par simplemma (MIT).
"""
import itertools
import re

import numpy as np
import simplemma

from .vocab import fold, normalize

POS_KEPT = {"n", "v", "a", "adv"}
FREQUENT_RANK = 25_000  # en dessous : on garde même les mots absents du dictionnaire (« beaucoup », « taux »)
WORD_PATTERN = re.compile(r"^[a-zàâäçéèêëîïôöùûüÿœæ]{2,}$")  # pas de chiffre, tiret, apostrophe ni espace


def read_tokens(path: str, keep_vectors: bool = True):
    """Génère (rang de fréquence, mot, vecteur ou None) dans l'ordre du fichier."""
    with open(path, "rb") as handle:
        count, dimension = (int(value) for value in handle.readline().split())
        for rank in range(count):
            word = bytearray()
            while True:
                char = handle.read(1)
                if char in (b" ", b""):
                    break
                if char != b"\n":
                    word += char
            raw = handle.read(4 * dimension)
            vector = np.frombuffer(raw, dtype="<f4").copy() if keep_vectors else None
            yield rank, word.decode("utf-8", errors="ignore"), vector


# Particules grammaticales sans valeur de jeu. Les autres mots très fréquents (monde, temps, vie, être, faire…) SONT jouables.
STOP_WORDS = frozenset(
    "ne je tu il elle on nous vous ils elles me te se ma mon ta ton sa son notre votre leur la le les un une des du de au aux "
    "ce cet cette ces qui que quoi dont où et ou mais donc or ni car si en y".split()
)


def select_lemmas(path: str, size: int, blocklist: set[str], minimum_rank: int = 0, forced: frozenset = frozenset()):
    """Premiers `size` lemmes jouables (nom, verbe, adjectif, adverbe), connus du dictionnaire.

    `forced` : lemmes à garder quoi qu'il arrive (vocabulaire déjà publié, mots du jour candidats) : append-only.
    Renvoie [(lemme, pos, rang_frequence, vecteur, connu_du_dictionnaire)] ; un lemme présent sous plusieurs pos garde la plus fréquente.
    """
    kept: dict[str, tuple] = {}
    for rank, token, vector in read_tokens(path):
        if rank < minimum_rank or "_" not in token:
            continue
        lemma, pos = token.rsplit("_", 1)
        lemma = normalize(lemma)
        if not lemma or pos not in POS_KEPT or lemma in kept:
            continue
        if lemma in STOP_WORDS and lemma not in forced:
            continue
        known = simplemma.is_known(lemma, lang="fr")
        if lemma not in forced:
            if lemma in blocklist or not WORD_PATTERN.match(lemma) or len(kept) >= size:
                continue
            if not known and rank >= FREQUENT_RANK:
                continue  # rare ET inconnu du dictionnaire : bruit (sigles, fautes, noms propres rares)
        kept[lemma] = (lemma, pos, rank, vector, known)
    return list(kept.values())


# Couples relus et refusés à la main (Angela) : formes inexistantes ou lecture verbale ignorée
REJECTED_FORMS = frozenset(
    'chaques qualitées canale ratione fallois collationes pucee carrieres rejete ajoutes lancee classees verdière usitée douzes remords journals purines courrière courrières canarie conclusiones monsieurs affilées sermones manufacturers'.split()
)
RULE_RANK_LIMIT = 150_000  # au-delà, trop de bruit : seul simplemma décide
RULE_LEMMA_RANK_LIMIT = 40_000  # le lemme cible doit lui-même être un mot courant (pas « dre », « jl », « pda »)

# (fin de la forme, remplacement, catégories admises pour le lemme) : participes et féminins que simplemma ne relie pas
FORM_RULES = (
    ("ées", "er", {"v"}), ("ée", "er", {"v"}), ("és", "er", {"v"}),  # menées → mener
    ("ées", "é", {"a"}), ("ée", "é", {"a"}), ("és", "é", {"a"}),  # fixée → fixé
    ("ies", "ir", {"v"}), ("ie", "ir", {"v"}), ("is", "ir", {"v"}),  # définies → définir
    ("ies", "i", {"a", "n"}), ("ie", "i", {"a", "n"}), ("is", "i", {"a", "n"}),  # définies → défini
    ("ues", "u", {"a", "n"}), ("ue", "u", {"a", "n"}), ("us", "u", {"a", "n"}),  # connues → connu
    ("ales", "al", {"a", "n"}), ("ale", "al", {"a", "n"}),  # nationale → national
    ("ives", "if", {"a", "n"}), ("ive", "if", {"a", "n"}),  # active → actif
    ("ières", "ier", {"a", "n"}), ("ière", "ier", {"a", "n"}),  # première → premier
    ("ennes", "en", {"a", "n"}), ("enne", "en", {"a", "n"}), ("iennes", "ien", {"a", "n"}), ("ienne", "ien", {"a", "n"}),
    ("elles", "el", {"a", "n"}), ("elle", "el", {"a", "n"}),  # nouvelle → nouvel (rare), éternelle → éternel
    ("euses", "eux", {"a", "n"}), ("euse", "eux", {"a", "n"}), ("euses", "eur", {"a", "n"}), ("euse", "eur", {"a", "n"}),
    ("trices", "teur", {"a", "n"}), ("trice", "teur", {"a", "n"}),
    ("es", "", {"a", "n"}), ("e", "", {"a"}), ("s", "", {"a", "n"}),  # étroites → étroit, faits → fait
)


def rule_lemma(form: str, lemmas: set[str], pos_of: dict[str, tuple[str, int]]) -> str | None:
    """Lemme d'une forme que simplemma n'a pas reliée : première règle dont le résultat est un lemme jouable de la bonne catégorie."""
    for ending, replacement, allowed in FORM_RULES:
        if not form.endswith(ending) or len(form) - len(ending) < 2:
            continue
        candidate = form[: len(form) - len(ending)] + replacement
        known = pos_of.get(candidate)
        if form in REJECTED_FORMS or form.endswith(("ones", "iones")):  # calques espagnols/latins
            return None
        if candidate != form and candidate in lemmas and known and known[0] in allowed and len(candidate) >= 4 and known[1] < RULE_LEMMA_RANK_LIMIT and simplemma.is_known(candidate, lang="fr"):
            return candidate
    return None


def select_forms(path: str, lemmas: set[str], limit_rank: int = 400_000, pos_of: dict[str, tuple[str, int]] | None = None):
    """Couples (forme, lemme) pour les formes du modèle non lemmatisé dont le lemme est jouable.

    Premier passage : simplemma. Second passage (si `pos_of` est fourni) : règles de participes/féminins pour les formes
    fréquentes que simplemma n'a pas reliées. Renvoie (couples, couples issus des règles).
    """
    pairs: dict[str, str] = {}
    by_rule: dict[str, str] = {}
    # Meilleure graphie (rang le plus fréquent) de chaque clé sans accent : « lancee » perd contre « lancée »
    best_spelling: dict[str, tuple[int, str]] = {}
    if pos_of is not None:
        for rank, form, _ in read_tokens(path, keep_vectors=False):
            form = normalize(form)
            if rank < RULE_RANK_LIMIT and WORD_PATTERN.match(form):
                key = fold(form)
                if key not in best_spelling:
                    best_spelling[key] = (rank, form)
    for rank, form, _ in read_tokens(path, keep_vectors=False):
        form = normalize(form)
        if rank >= limit_rank or form in lemmas or not WORD_PATTERN.match(form):
            continue
        lemma = normalize(simplemma.lemmatize(form, lang="fr"))
        if lemma in lemmas and lemma != form:
            pairs.setdefault(form, lemma)
        elif pos_of is not None and rank < RULE_RANK_LIMIT and form not in pairs and best_spelling[fold(form)][1] == form:
            ruled = rule_lemma(form, lemmas, pos_of)
            if ruled:
                pairs.setdefault(form, ruled)
                by_rule[form] = ruled
    return sorted(pairs.items()), sorted(by_rule.items())


def compound_forms(path: str, lemmas, limit_rank: int = 400_000) -> list[tuple[str, str]]:
    """Pluriels des composés à tiret (« grands-mères », « chauves-souris ») : couples (forme, lemme).

    Le lemmatiseur et WORD_PATTERN ignorent les tirets ; pour chaque lemme à tiret on génère les variantes où un sous-ensemble de morceaux
    prend un « s », et on ne garde que celles attestées dans le modèle non lemmatisé (rang < limit_rank). Un lemme n'a jamais d'apostrophe.
    """
    wanted: dict[str, str] = {}
    for lemma in lemmas:
        parts = lemma.split("-")
        if len(parts) < 2 or "'" in lemma:
            continue
        pluralisable = [index for index, part in enumerate(parts) if not part.endswith(("s", "x", "z"))]
        for size in range(1, len(pluralisable) + 1):
            for chosen in itertools.combinations(pluralisable, size):
                variant = "-".join(part + "s" if index in chosen else part for index, part in enumerate(parts))
                wanted.setdefault(variant, lemma)
    found: dict[str, str] = {}
    for rank, form, _ in read_tokens(path, keep_vectors=False):
        if rank >= limit_rank:
            break
        form = normalize(form)
        if form in wanted and form not in lemmas:
            found[form] = wanted[form]
    return sorted(found.items())
