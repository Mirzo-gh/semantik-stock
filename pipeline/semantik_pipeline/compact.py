"""Encodages compacts des fichiers de vocabulaire (DATA_CONTRACT §2b), dérivés à l'identique des fichiers texte d'origine.

Le fichier d'origine (lemmas.<hash>.txt / forms.<hash>.txt) reste la référence : index, hash et contenu ne changent pas.
Les versions compactes sont une FONCTION PURE de l'original, servies sous le même hash avec un suffixe :
  lemmas.<hash>.fc.txt  « front-coded »  : ligne 1 `#fc1`, puis une ligne par lemme (index = n° de ligne après l'en-tête)
                        = 1 caractère base 36 (longueur du préfixe commun avec le lemme précédent, plafonnée à 35) + suffixe.
                        Un lemme neutralisé (`~mot`) devient la ligne `0~` (index conservé, jamais jouable) et ne sert pas de précédent.
  forms.<hash>.sx.txt   « suffixes »     : ligne 1 `#sx1`, puis une ligne par lemme ayant des formes :
                        `<écart d'index depuis la ligne précédente>;<coupe><suffixe>,<coupe><suffixe>,…`
                        forme = lemme privé de `coupe` derniers caractères + `suffixe`.
Décodage de référence côté client : app/src/lib/game/data.ts.
"""

DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
LEMMAS_HEADER = "#fc1"
FORMS_HEADER = "#sx1"
TOMBSTONE = "~"


def _check_bmp(word: str) -> None:
    if any(ord(char) > 0xFFFF for char in word):
        raise ValueError(f"caractère hors plan de base : {word!r}")  # JS compte en unités UTF-16


def encode_lemmas(lemmas: list[str]) -> str:
    lines, previous = [LEMMAS_HEADER], ""
    for word in lemmas:
        if word.startswith(TOMBSTONE):
            lines.append("0" + TOMBSTONE)
            continue
        _check_bmp(word)
        shared = 0
        limit = min(len(word), len(previous), 35)
        while shared < limit and word[shared] == previous[shared]:
            shared += 1
        lines.append(DIGITS[shared] + word[shared:])
        previous = word
    return "\n".join(lines) + "\n"


def decode_lemmas(text: str) -> list[str]:
    lines = text.split("\n")
    if lines[0] != LEMMAS_HEADER:
        raise ValueError("en-tête #fc1 attendu")
    lemmas, previous = [], ""
    for line in lines[1:]:
        if line == "":
            continue
        if line == "0" + TOMBSTONE:
            lemmas.append(TOMBSTONE)
            continue
        word = previous[: DIGITS.index(line[0])] + line[1:]
        lemmas.append(word)
        previous = word
    return lemmas


def encode_forms(pairs: list[tuple[str, int]], lemmas: list[str]) -> str:
    by_lemma: dict[int, list[str]] = {}
    for form, index in pairs:
        if not form or any(char.isdigit() or char in ",;" for char in form):
            raise ValueError(f"forme non encodable : {form!r}")
        _check_bmp(form)
        by_lemma.setdefault(index, []).append(form)
    lines, last = [FORMS_HEADER], 0
    for index in sorted(by_lemma):
        lemma, items = lemmas[index], []
        for form in sorted(by_lemma[index]):
            shared = 0
            limit = min(len(form), len(lemma))
            while shared < limit and form[shared] == lemma[shared]:
                shared += 1
            items.append(f"{len(lemma) - shared}{form[shared:]}")
        lines.append(f"{index - last};" + ",".join(items))
        last = index
    return "\n".join(lines) + "\n"


def decode_forms(text: str, lemmas: list[str]) -> list[tuple[str, int]]:
    lines = text.split("\n")
    if lines[0] != FORMS_HEADER:
        raise ValueError("en-tête #sx1 attendu")
    pairs, index = [], 0
    for line in lines[1:]:
        if line == "":
            continue
        delta, items = line.split(";", 1)
        index += int(delta)
        lemma = lemmas[index]
        for item in items.split(","):
            digits = 0
            while digits < len(item) and item[digits].isdigit():
                digits += 1
            cut = int(item[:digits])
            pairs.append((lemma[: len(lemma) - cut] + item[digits:], index))
    return pairs
