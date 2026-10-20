"""Score mélangé frWac + fastText, avec graines de sens (étude hive/work/semantik/algo/ETUDE.md).

- Chaque modèle donne une similarité cosinus au mot du jour, centrée-réduite (z-score) sur le vocabulaire : les deux échelles
  sont ainsi comparables ; la moyenne des z est le score mélangé. Un lemme sans vecteur fastText garde le z de frWac seul.
- Graines de sens (pipeline/senses.tsv) : pour un mot polysémique, chaque sens est un groupe de mots-exemples ; le cosinus d'un lemme
  devient le maximum entre sa similarité au mot du jour et SENSE_WEIGHT × sa similarité à la moyenne du groupe (pour une graine : à la moyenne des autres graines).
- Les vecteurs fastText sont précalculés hors CI (precompute_fasttext.py) et ne sont JAMAIS publiés.
- Le résultat est une « similarité » où le mot du jour vaut 1 (transformation affine du z-score : les °C, ancrés sur des quantiles,
  n'en dépendent pas), ce qui permet de réutiliser scoring.stored_scores_from_similarities tel quel.
"""
import numpy as np

# V2 (étude VALIDATION-V2) : associations humaines JeuxDeMots (CC0) et pénalité des mots rares
JDM_BETA = 3.0  # bonus maximal, en écarts-types du score mélangé, pour une association humaine forte
JDM_REFERENCE_WEIGHT = 500.0  # poids JeuxDeMots à partir duquel une association compte pleinement
RARE_START_RANK = 30_000  # rang de fréquence frWac au-delà duquel un lemme est pénalisé
RARE_LAMBDA = 6.0  # pénalité, en écarts-types, par décade de rang au-delà de RARE_START_RANK
FILTER_PENALTIES = {"nom_propre": 8.0, "mot_outil": 8.0, "variante": 3.0, "intrus": 8.0, "sigle_marque": 8.0}  # écarts-types retirés au score (jamais pour le mot du jour)

EXCLUSION_PENALTY = 12.0  # écarts-types retirés à un voisin exclu à la relecture (pipeline/exclusions.tsv, privé) : il sort du top 1000 sans autre effet
# Variante V3 (étude PROPRE-DIFFICULTE, NON décidée) : valeurs par défaut = calcul actuel ; la relecture se prépare avec les deux
CURRENT_SETTINGS = {"fasttext_weight": 0.5, "jdm_beta": JDM_BETA, "unsupported_penalty": 0.0}
V3_SETTINGS = {"fasttext_weight": 0.35, "jdm_beta": 4.5, "unsupported_penalty": 1.0}
UNSUPPORTED_MIN_ASSOCIATIONS = 50  # la pénalité V3 des lemmes sans association humaine ne vaut que si le mot en a au moins autant

SENSE_WEIGHT = 0.8  # réglé sur 4 mots puis par test à l'aveugle (Angela) : à 0,6 le sens principal gagne trop, à 0,9 un seul sens secondaire remplit le haut de liste


def unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    return matrix / norms[:, None]


def load_fasttext(path: str) -> tuple[list[str], np.ndarray, set[str]]:
    """(noms de lemmes, vecteurs float16, lemmes cherchés mais absents de fastText) d'un fichier précalculé."""
    with np.load(path, allow_pickle=False) as archive:
        return archive["names"].tolist(), archive["vectors"], set(archive["absent"].tolist())


def align_fasttext(names: list[str], vectors: np.ndarray, lemmas: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Matrice float32 alignée sur `lemmas` (les « ~ » de neutralisation ignorés) et masque des lemmes présents."""
    row_of = {name: row for row, name in enumerate(names)}
    aligned = np.zeros((len(lemmas), vectors.shape[1]), dtype=np.float32)
    present = np.zeros(len(lemmas), dtype=bool)
    for index, lemma in enumerate(lemmas):
        row = row_of.get(lemma.removeprefix("~"))
        if row is not None:
            aligned[index] = vectors[row]
            present[index] = True
    return aligned, present


def parse_senses(text: str, lemmas: list[str]) -> dict[str, list[list[int]]]:
    """pipeline/senses.tsv : `mot<TAB>graine graine | graine graine` (un groupe par sens). Renvoie {mot: [[index de graine]]}.

    Échoue (jamais en silence) sur un mot ou une graine absents du vocabulaire, ou un groupe vide.
    """
    index_of = {lemma: index for index, lemma in enumerate(lemmas)}
    senses: dict[str, list[list[int]]] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        word, groups_text = line.split("\t", 1)
        if word not in index_of:
            raise SystemExit(f"senses.tsv : mot absent du vocabulaire : {word}")
        groups = []
        for group_text in groups_text.split("|"):
            seeds = group_text.split()
            unknown = [seed for seed in seeds if seed not in index_of]
            if not seeds or unknown:
                raise SystemExit(f"senses.tsv : {word} : groupe vide ou graines absentes du vocabulaire {unknown}")
            groups.append([index_of[seed] for seed in seeds])
        senses[word] = groups
    return senses


def parse_associations(text: str, lemmas: list[str]) -> dict[str, np.ndarray]:
    """pipeline/jdm_associations.tsv : `mot<TAB>lemme:poids lemme:poids …` (JeuxDeMots, r_associated, CC0). Renvoie {mot: poids par lemme}.

    Les lemmes inconnus du vocabulaire courant sont ignorés (le fichier reste valable quand le vocabulaire change).
    """
    index_of = {lemma: index for index, lemma in enumerate(lemmas)}
    associations: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        word, _, pairs = line.partition("\t")
        weights = np.zeros(len(lemmas))
        for pair in pairs.split():
            lemma, _, weight = pair.rpartition(":")
            if lemma in index_of:
                weights[index_of[lemma]] = float(weight)
        associations[word] = weights
    return associations


def parse_filter(text: str, lemmas: list[str]) -> np.ndarray:
    """pipeline/filter_lemmas.tsv : `lemme<TAB>motif<TAB>forme principale` (noms propres, mots-outils, variantes d'orthographe).

    Renvoie la pénalité de chaque lemme (0 = aucun filtre). Un lemme inconnu du vocabulaire est ignoré.
    """
    index_of = {lemma: index for index, lemma in enumerate(lemmas)}
    penalties = np.zeros(len(lemmas))
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        lemma, motif = line.split("\t")[:2]
        if lemma in index_of:
            penalties[index_of[lemma]] = FILTER_PENALTIES[motif]
    return penalties


def parse_exclusions(text: str, lemmas: list[str]) -> dict[str, np.ndarray]:
    """pipeline/exclusions.tsv (PRIVÉ, jamais dans le dépôt) : `mot<TAB>lemme lemme …` -> {mot: pénalité de chaque lemme}.

    Un lemme inconnu du vocabulaire est ignoré (le fichier reste valable quand le vocabulaire change) ; une ligne vide ou « # » est sautée.
    """
    index_of = {lemma: index for index, lemma in enumerate(lemmas)}
    penalties: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        word, _, excluded = line.partition("\t")
        vector = penalties.setdefault(word, np.zeros(len(lemmas)))
        for lemma in excluded.split():
            if lemma in index_of:
                vector[index_of[lemma]] = EXCLUSION_PENALTY
    return penalties


def read_frequency_ranks(path: str, lemmas: list[str]) -> np.ndarray:
    """Rang de fréquence frWac de chaque lemme (data-work/pos.tsv : lemme, catégorie, rang, connu) ; 10^6 si absent."""
    rank_of = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            rank_of[fields[0]] = float(fields[2])
    return np.array([rank_of.get(lemma.removeprefix("~"), 1e6) for lemma in lemmas])


def model_scores(unit_vectors: np.ndarray, target_index: int, seed_groups: list[list[int]], valid: np.ndarray) -> np.ndarray:
    """z-score (sur les lemmes `valid`) du cosinus maximal entre le mot du jour et ses sens ; 0 hors `valid`."""
    similarities = unit_vectors @ unit_vectors[target_index]
    for seeds in seed_groups:
        centroid = unit_vectors[seeds].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            boosted = SENSE_WEIGHT * (unit_vectors @ (centroid / norm))
            # Une graine est forcément proche de la moyenne de son groupe qui la contient : on la compare à la moyenne des AUTRES graines
            # (sinon les graines s'affichent elles-mêmes en tête)
            for seed in seeds:
                others = unit_vectors[[other for other in seeds if other != seed]].sum(axis=0) if len(seeds) > 1 else None
                if others is not None and np.linalg.norm(others) > 0:
                    boosted[seed] = SENSE_WEIGHT * float(unit_vectors[seed] @ (others / np.linalg.norm(others)))
            similarities = np.maximum(similarities, boosted)
    mean, spread = similarities[valid].mean(), similarities[valid].std()
    scores = np.zeros(len(similarities))
    scores[valid] = (similarities[valid] - mean) / spread
    return scores


def blended_scores(frwac: np.ndarray, fasttext: np.ndarray, fasttext_present: np.ndarray,
                         target_index: int, seed_groups: list[list[int]] | None = None,
                         associations: np.ndarray | None = None, frequency_ranks: np.ndarray | None = None,
                  filter_penalties: np.ndarray | None = None, exclusion_penalties: np.ndarray | None = None,
                  fasttext_weight: float = 0.5, jdm_beta: float = JDM_BETA, unsupported_penalty: float = 0.0) -> np.ndarray:
    """Score mélangé brut (z-score moyen + bonus/pénalités), le mot du jour est le maximum.

    `associations` : poids JeuxDeMots (0 = aucune) de chaque lemme avec le mot du jour ; `frequency_ranks` : rang de fréquence frWac de chaque lemme.
    """
    seed_groups = seed_groups or []
    if not fasttext_present[target_index]:
        raise SystemExit("mot du jour sans vecteur fastText : relancer pipeline/precompute_fasttext.py")
    everywhere = np.ones(len(frwac), dtype=bool)
    z_frwac = model_scores(unit(frwac), target_index, seed_groups, everywhere)
    seeds_present = [[seed for seed in seeds if fasttext_present[seed]] for seeds in seed_groups]
    z_fasttext = model_scores(unit(fasttext), target_index, [seeds for seeds in seeds_present if seeds], fasttext_present)
    # Poids 0,5 : expression d'origine, à l'identique près (les jours déjà calculés ne doivent pas bouger d'une unité)
    mixed = (z_frwac + z_fasttext) / 2 if fasttext_weight == 0.5 else (1 - fasttext_weight) * z_frwac + fasttext_weight * z_fasttext
    blended = np.where(fasttext_present, mixed, z_frwac)
    if associations is not None:
        blended = blended + jdm_beta * np.minimum(1.0, np.log1p(associations) / np.log1p(JDM_REFERENCE_WEIGHT))
        if unsupported_penalty and (associations > 0).sum() >= UNSUPPORTED_MIN_ASSOCIATIONS:
            blended = blended - unsupported_penalty * (associations == 0)
    if frequency_ranks is not None:
        blended = blended - RARE_LAMBDA * np.maximum(0.0, np.log10(np.maximum(frequency_ranks, 1) / RARE_START_RANK))
    if filter_penalties is not None:
        penalties = filter_penalties.copy()
        penalties[target_index] = 0.0  # le mot du jour n'est jamais filtré
        blended = blended - penalties
    if exclusion_penalties is not None:
        excluded = exclusion_penalties.copy()
        excluded[target_index] = 0.0
        blended = blended - excluded
    blended[target_index] = max(blended[target_index], np.delete(blended, target_index).max() + 0.5)  # le mot du jour reste le maximum
    return blended


def blended_similarities(*arguments, **keywords) -> np.ndarray:
    """Pseudo-similarité mélangée (le mot du jour vaut 1, c'est le maximum), mêmes arguments que blended_scores."""
    blended = blended_scores(*arguments, **keywords)
    target_score = blended[arguments[3] if len(arguments) > 3 else keywords["target_index"]]
    spread = max(target_score - blended.min(), 1e-9)
    return 1.0 - (target_score - blended) / spread
