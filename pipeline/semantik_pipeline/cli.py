"""Génère les fichiers de données.

Vocabulaire (rare, quand on change de version) :
  python -m semantik_pipeline.cli vocab --model frwac.bin --lexique Lexique383.tsv --out data
Puzzle du jour (nuit, avec SEMANTIK_MASTER_KEY) :
  python -m semantik_pipeline.cli daily --data data --date 2026-10-07 --number 2 --candidates candidates.txt
"""
import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np

from . import blend as blend_module, change as change_module, daily as daily_module, difficulty as difficulty_module, start_word as start_word_module, vocab as vocab_module
from .w2v import load_word2vec_binary


def read_lines(path: str) -> list[str]:
    return [line.strip() for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def command_vocab(args) -> None:
    rows = vocab_module.read_lexique(args.lexique)
    model_words = set(json.loads(pathlib.Path(args.model_words).read_text())) if args.model_words else None
    vectors = load_word2vec_binary(args.model, wanted=model_words)
    blocklist = {vocab_module.normalize(word) for word in read_lines(args.blocklist)} if args.blocklist else set()
    lemmas, pairs = vocab_module.build_vocab(rows, set(vectors), args.size, blocklist)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lemma_text = vocab_module.render_lemmas(lemmas)
    forms_text = vocab_module.render_forms(pairs, lemmas)
    (out / f"lemmas.{vocab_module.content_hash(lemma_text)}.txt").write_text(lemma_text, encoding="utf-8")
    (out / f"forms.{vocab_module.content_hash(forms_text)}.txt").write_text(forms_text, encoding="utf-8")
    np.save(out / "vectors.npy", np.stack([vectors[lemma] for lemma in lemmas]))  # cache CI, jamais publié
    print(f"{len(lemmas)} lemmes, {len(pairs)} formes")


def read_pos_classes(path: str) -> dict[str, int]:
    """Table lemme<TAB>classe (0 nom, 1 verbe, 2 adjectif, 3 adverbe, 4 autre) ; un lemme absent n'a pas d'indice « nature »."""
    classes: dict[str, int] = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        lemma, _, label = line.partition("\t")
        if lemma and label.strip().isdigit() and 0 <= int(label) <= 4:
            classes[lemma] = int(label)
    return classes


def read_forced_words(path: str) -> dict[int, str]:
    """Fichier PRIVÉ numéro<TAB>mot (décision admin : mot choisi pour un jour futur). Lignes vides et « # » ignorées."""
    forced: dict[int, str] = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        number, _, word = line.partition("\t")
        if not number.strip().isdigit() or not word.strip():
            sys.exit("--force-word-file : ligne illisible (attendu : numéro<TAB>mot)")
        forced[int(number)] = word.strip()
    return forced


def draw_pools(args, candidates: list[str]) -> tuple[list[str], list[str], int]:
    """(tirage avant le filtre « mots communs », tirage filtré, premier jour du filtre) : le filtre ne touche que les jours >= le troisième nombre."""
    if not args.difficulty:
        return candidates, candidates, 1
    base = difficulty_module.keep_candidates(candidates, difficulty_module.parse_difficulty(pathlib.Path(args.difficulty).read_text(encoding="utf-8")), args.drop_classes)
    excluded = difficulty_module.read_draw_exclusions(args.difficulty)
    return base, [word for word in base if word not in excluded], difficulty_module.read_draw_exclusions_from(args.difficulty)


def check_forced_word(number: int, word: str, history: list[str], lemma_set: set[str], draw_candidates: list[str]) -> None:
    """Un mot imposé doit être jouable (voir change.word_problems). Les messages ne contiennent que le numéro du jour, jamais le mot."""
    for code in change_module.word_problems(word, history, lemma_set, draw_candidates):
        sys.exit(f"n°{number} : mot imposé refusé ({change_module.REFUSAL_TEXTS[code]})")


def command_daily(args) -> None:
    master_key = os.environ.get("SEMANTIK_MASTER_KEY", "").encode()
    if not master_key:
        sys.exit("SEMANTIK_MASTER_KEY absente")
    data = pathlib.Path(args.data)
    lemma_file = next(data.glob("lemmas.*.txt"))
    forms_file = next(data.glob("forms.*.txt"))
    lemmas = lemma_file.read_text(encoding="utf-8").splitlines()
    lemma_set = set(lemmas)
    candidates = read_lines(args.candidates)
    pos_classes = read_pos_classes(args.pos) if args.pos else {}
    missing = [word for word in candidates if word not in lemma_set]
    if missing:  # ne jamais filtrer en silence : cela changerait les tirages
        sys.exit(f"candidats absents du vocabulaire : {missing[:10]}")
    vectors = np.load(data / "vectors.npy")
    # Score mélangé frWac + fastText + graines de sens, à partir du jour --blend-from (jours antérieurs : calcul historique, inchangé)
    blending = args.blend_from is not None and args.number >= args.blend_from
    fasttext = fasttext_present = None
    senses: dict[str, list[list[int]]] = {}
    associations: dict = {}
    frequency_ranks = None
    filter_penalties = None
    if blending:
        if not args.fasttext or not pathlib.Path(args.fasttext).exists():
            sys.exit("--blend-from demande les vecteurs fastText précalculés (--fasttext ft.npz, voir pipeline/precompute_fasttext.py)")
        names, fasttext_vectors, _ = blend_module.load_fasttext(args.fasttext)
        fasttext, fasttext_present = blend_module.align_fasttext(names, fasttext_vectors, lemmas)
        # Un futur mot du jour sans vecteur fastText ferait échouer un jour précis : on échoue tout de suite, pour tous les candidats
        uncovered = [word for word in candidates if not fasttext_present[lemmas.index(word)]]
        if uncovered:
            sys.exit(f"candidats sans vecteur fastText (relancer pipeline/precompute_fasttext.py) : {uncovered[:10]}")
        if args.jdm:
            associations = blend_module.parse_associations(pathlib.Path(args.jdm).read_text(encoding="utf-8"), [lemma.removeprefix("~") for lemma in lemmas])
            pos_path = data / "pos.tsv"
            frequency_ranks = blend_module.read_frequency_ranks(str(pos_path), lemmas) if pos_path.exists() else None
        if args.filter:
            filter_penalties = blend_module.parse_filter(pathlib.Path(args.filter).read_text(encoding="utf-8"), [lemma.removeprefix("~") for lemma in lemmas])
        if args.senses:
            senses = blend_module.parse_senses(pathlib.Path(args.senses).read_text(encoding="utf-8"), [lemma.removeprefix("~") for lemma in lemmas])
    # Nettoyage du vocabulaire (jours >= --cleanup-from seulement, jamais un jour déjà calculé) : les lemmes neutralisés « ~ » (injouables) et les prénoms
    # de pipeline/filter_names.tsv sortent du classement comme le filtre ; ils ne peuvent plus occuper une place du top 1000
    cleanup_penalties = None
    if args.cleanup_from is not None and filter_penalties is not None:
        cleanup_penalties = np.maximum(filter_penalties, np.array([blend_module.FILTER_PENALTIES["intrus"] if lemma.startswith("~") else 0.0 for lemma in lemmas]))
        if args.names:
            cleanup_penalties = np.maximum(cleanup_penalties, blend_module.parse_filter(pathlib.Path(args.names).read_text(encoding="utf-8"), [lemma.removeprefix("~") for lemma in lemmas]))
    exclusions: dict = {}
    if args.exclusions:
        # Relecture des voisins par les agents (fichier PRIVÉ, absent = aucune exclusion) : ne s'applique qu'aux jours calculés maintenant
        exclusions = blend_module.parse_exclusions(pathlib.Path(args.exclusions).read_text(encoding="utf-8"), [lemma.removeprefix("~") for lemma in lemmas])
    # Difficulté : les classes retirées (ex. « dx ») ne sont plus tirées. Les jours déjà publiés gardent leur mot (historique relu dans les fichiers),
    # un jour épinglé (--pinned) aussi : seul le tirage des jours encore à faire change.
    base_candidates, filtered_candidates, exclusions_start = draw_pools(args, candidates)
    if not filtered_candidates:
        sys.exit("--drop-classes retire tous les candidats")
    # Mot de départ à 30 °C (jours >= --start-from) : fréquence frWac de data/pos.tsv, exclusions (noms propres, liste noire, filtre)
    start_ranks = start_excluded = None
    if args.start_from is not None:
        pos_path = data / "pos.tsv"
        if not pos_path.exists():
            sys.exit("--start-from demande data/pos.tsv (rang de fréquence)")
        start_ranks = blend_module.read_frequency_ranks(str(pos_path), lemmas)
        start_excluded = start_word_module.read_excluded_words(args.start_exclude or [])
    forced_words = read_forced_words(args.force_word_file) if args.force_word_file else {}
    target = data / "daily"
    target.mkdir(exist_ok=True)
    last_day = datetime.date.fromisoformat(args.date)
    # Planning à état : on rejoue l'historique des jours 1..n-1 (fichiers déjà publiés, jamais régénérés)
    # et on ne tire que les jours manquants, dans l'ordre. Le vocabulaire est append-only : les index anciens restent valables.
    history: list[str] = []
    for number in range(1, args.number + 1):
        path = target / f"{number}.json"
        if path.exists():
            history.append(lemmas[daily_module.decode_target_index(json.loads(path.read_text()))])
            continue
        pinned_file = pathlib.Path(args.pinned, "daily", f"{number}.json") if args.pinned else None
        if number in forced_words:
            # Mot imposé (admin, jour futur) : prime sur l'épingle et sur le tirage ; l'historique des jours suivants en tient compte
            word = forced_words[number]
            check_forced_word(number, word, history, lemma_set, filtered_candidates if number >= exclusions_start else base_candidates)
        elif pinned_file is not None and pinned_file.exists():
            # Jour régénéré sur un nouveau vocabulaire : on garde EXACTEMENT le mot déjà publié (l'index reste valable : vocabulaire append-only)
            word = lemmas[daily_module.decode_target_index(json.loads(pinned_file.read_text()))]
        else:
            word = daily_module.next_word(master_key, history, filtered_candidates if number >= exclusions_start else base_candidates)
        history.append(word)
        day = (last_day - datetime.timedelta(days=args.number - number)).isoformat()
        word_index = lemmas.index(word)
        similarities = None
        if blending and number >= args.blend_from:
            similarities = blend_module.blended_similarities(vectors, fasttext, fasttext_present, word_index, senses.get(word), associations.get(word), frequency_ranks, cleanup_penalties if cleanup_penalties is not None and number >= args.cleanup_from else filter_penalties, exclusions.get(word), **(blend_module.V3_SETTINGS if args.v3_from is not None and number >= args.v3_from else blend_module.CURRENT_SETTINGS))
        start_picker = None
        if args.start_from is not None and number >= args.start_from:
            def start_picker(scores, top, day=day, word_index=word_index):
                index, half_width, count = start_word_module.choose_start(master_key, day, lemmas, scores, top, word_index, start_ranks, start_excluded)
                print(f"n°{number} départ : {count} candidats (bande ±{half_width / 100:g} °C)" + ("" if index is not None else ", AUCUN"), file=sys.stderr)  # jamais le mot
                return index
        puzzle = daily_module.build_daily(
            vectors, word_index, number, day,
            lemma_file.name.split(".")[1], forms_file.name.split(".")[1],
            daily_module.derive_seed(master_key, day), similarities, pos_classes.get(word), start_picker,
        )
        path.write_text(daily_module.dump(puzzle))


def command_plan(args) -> None:
    """Les N prochains mots non encore calculés, d'après l'historique des fichiers daily déjà là. PRIVÉ : à rediriger vers un fichier hors dépôt."""
    master_key = os.environ.get("SEMANTIK_MASTER_KEY", "").encode()
    if not master_key:
        sys.exit("SEMANTIK_MASTER_KEY absente")
    data = pathlib.Path(args.data)
    lemmas = next(data.glob("lemmas.*.txt")).read_text(encoding="utf-8").splitlines()
    candidates = read_lines(args.candidates)
    base_candidates, filtered_candidates, exclusions_start = draw_pools(args, candidates)
    history: list[str] = []
    number = 1
    while (data / "daily" / f"{number}.json").exists():
        history.append(lemmas[daily_module.decode_target_index(json.loads((data / "daily" / f"{number}.json").read_text()))])
        number += 1
    first = number
    rows = []
    for _ in range(args.days):
        word = daily_module.next_word(master_key, history, filtered_candidates if number >= exclusions_start else base_candidates)
        history.append(word)
        rows.append(f"{number}\t{word}")
        number += 1
    text = "\n".join(rows) + "\n"
    if args.out == "-":
        sys.stdout.write(text)  # le plan EST la sortie standard (à rediriger vers un fichier privé, jamais vers un journal)
        print(f"{args.days} jours (n°{first} à {number - 1})", file=sys.stderr)
    else:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print(f"{args.days} jours (n°{first} à {number - 1}) -> fichier privé, aucun mot affiché")


def command_change(args) -> None:
    """Contrôle et résolution d'une demande de changement de mot (admin). Écrit {codes, word, neighbours_unreviewed} dans --out : fichier PRIVÉ, jamais affiché."""
    master_key = os.environ.get("SEMANTIK_MASTER_KEY", "").encode()
    if not master_key:
        sys.exit("SEMANTIK_MASTER_KEY absente")
    data = pathlib.Path(args.data)
    lemmas = next(data.glob("lemmas.*.txt")).read_text(encoding="utf-8").splitlines()
    candidates = read_lines(args.candidates)
    base_candidates, filtered_candidates, exclusions_start = draw_pools(args, candidates)
    draw_candidates = filtered_candidates if args.number >= exclusions_start else base_candidates
    try:
        request = json.loads(pathlib.Path(args.request).read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError
    except ValueError:
        request = {}
    daily_dir = data / "daily"
    history = change_module.history_words(daily_dir, lemmas, args.number)
    old_path = daily_dir / f"{args.number}.json"
    old_word = lemmas[daily_module.decode_target_index(json.loads(old_path.read_text()))] if old_path.exists() else ""
    exclusions_text = pathlib.Path(args.exclusions).read_text(encoding="utf-8") if args.exclusions and pathlib.Path(args.exclusions).exists() else None
    resolution = change_module.resolve(request, history, lemmas, draw_candidates, old_word, master_key, exclusions_text)
    pathlib.Path(args.out).write_text(json.dumps(resolution), encoding="utf-8")
    print(f"changement n°{args.number} : mode {request.get('mode', '?')}, {'refusé' if resolution['codes'] else 'accepté'} ({','.join(resolution['codes']) or 'ok'})")  # jamais le mot


def command_pool(args) -> None:
    """Export admin/pool.json : candidats de tirage (ceux sur lesquels `change` décide not_candidate) et mots de tous les jours 1..--last. Fichier PRIVÉ, jamais affiché."""
    data = pathlib.Path(args.data)
    lemmas = next(data.glob("lemmas.*.txt")).read_text(encoding="utf-8").splitlines()
    candidates = read_lines(args.candidates)
    base_candidates, draw_candidates, exclusions_start = draw_pools(args, candidates)
    days = change_module.history_words(data / "daily", lemmas, args.last + 1)
    if days is None:
        sys.exit("jour manquant : export du pool impossible")
    pool = {"v": 1, "at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "candidates": list(dict.fromkeys(draw_candidates)), "removed": [word for word in dict.fromkeys(base_candidates) if word not in set(draw_candidates)], "days": days}
    pathlib.Path(args.out).write_text(json.dumps(pool, ensure_ascii=False), encoding="utf-8")
    print(f"pool : {len(pool['candidates'])} candidats, {len(pool['removed'])} retirés, {len(days)} jours")  # des nombres seulement


def main() -> None:
    parser = argparse.ArgumentParser(prog="semantik_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    vocab = commands.add_parser("vocab")
    vocab.add_argument("--model", required=True)
    vocab.add_argument("--model-words")
    vocab.add_argument("--lexique", required=True)
    vocab.add_argument("--blocklist")
    vocab.add_argument("--size", type=int, default=40000)
    vocab.add_argument("--out", required=True)
    vocab.set_defaults(handler=command_vocab)
    daily = commands.add_parser("daily")
    daily.add_argument("--data", required=True)
    daily.add_argument("--date", required=True)
    daily.add_argument("--number", type=int, required=True)
    daily.add_argument("--candidates", required=True)
    daily.add_argument("--force-word-file", help="fichier PRIVÉ numéro<TAB>mot : mot imposé pour ces jours (futurs, pas encore générés) ; jamais affiché")
    daily.add_argument("--pinned", help="dossier contenant daily/<n>.json dont le mot secret doit être conservé (jours régénérés)")
    daily.add_argument("--blend-from", type=int, help="numéro du premier jour calculé avec le score mélangé (absent : calcul historique pour tous)")
    daily.add_argument("--fasttext", help="vecteurs fastText précalculés (ft.npz), requis avec --blend-from")
    daily.add_argument("--jdm", help="associations JeuxDeMots des candidats (pipeline/jdm_associations.tsv, CC0) : bonus d'association humaine + pénalité des mots rares (pos.tsv)")
    daily.add_argument("--filter", help="filtre de lemmes (pipeline/filter_lemmas.tsv) : noms propres, mots-outils, variantes d'orthographe")
    daily.add_argument("--pos", help="table lemme<TAB>classe de nature (indice « nature » : champ pos du fichier du jour) ; absente : pas de champ pos")
    daily.add_argument("--senses", help="graines de sens (pipeline/senses.tsv)")
    daily.add_argument("--cleanup-from", type=int, help="numéro du premier jour dont le classement écarte les lemmes neutralisés « ~ » et les prénoms (--names) ; absent : inchangé")
    daily.add_argument("--names", help="prénoms à écarter du classement (pipeline/filter_names.tsv, même format que filter_lemmas.tsv), avec --cleanup-from")
    daily.add_argument("--exclusions", help="exclusions de voisins relues par les agents (PRIVÉ : exclusions.tsv, jamais dans le dépôt)")
    daily.add_argument("--difficulty", help="pipeline/difficulty.tsv (mot, essais simulés, classe) ; avec --drop-classes")
    daily.add_argument("--drop-classes", default="dx", help="classes de difficulté retirées du tirage (défaut « dx » : difficile et non trouvé)")
    daily.add_argument("--v3-from", type=int, help="numéro du premier jour calculé avec le rééquilibrage V3 (fastText 0,35, JDM β 4,5, pénalité sans association) ; absent : calcul actuel. Bascule datée : les jours déjà publiés ne changent jamais")
    daily.add_argument("--start-from", type=int, help="numéro du premier jour qui reçoit un mot de départ à 30 °C (champ `start`) ; absent : aucun. Bascule datée, les jours déjà publiés ne changent jamais")
    daily.add_argument("--start-exclude", nargs="+", help="fichiers de mots jamais proposés comme départ (vocab_blocklist.txt, filter_lemmas.tsv, filter_names.tsv), avec --start-from")
    daily.set_defaults(handler=command_daily)
    change = commands.add_parser("change")
    change.add_argument("--data", required=True)
    change.add_argument("--number", type=int, required=True)
    change.add_argument("--request", required=True, help="demande JSON {mode: word|redraw|revert, word?, nonce?} (PRIVÉE)")
    change.add_argument("--candidates", required=True)
    change.add_argument("--difficulty")
    change.add_argument("--drop-classes", default="dx")
    change.add_argument("--exclusions")
    change.add_argument("--out", required=True, help="fichier PRIVÉ de résolution (contient le mot)")
    change.set_defaults(handler=command_change)
    pool = commands.add_parser("pool")
    pool.add_argument("--data", required=True)
    pool.add_argument("--last", type=int, required=True, help="dernier jour du stock")
    pool.add_argument("--candidates", required=True)
    pool.add_argument("--difficulty")
    pool.add_argument("--drop-classes", default="dx")
    pool.add_argument("--out", required=True, help="fichier PRIVÉ (contient les mots)")
    pool.set_defaults(handler=command_pool)
    plan = commands.add_parser("plan")
    plan.add_argument("--data", required=True)
    plan.add_argument("--candidates", required=True)
    plan.add_argument("--days", type=int, default=30)
    plan.add_argument("--out", default="-", help="fichier privé de sortie ; « - » (défaut) : sortie standard, format plan.tsv (numéro<TAB>mot)")
    plan.add_argument("--difficulty")
    plan.add_argument("--drop-classes", default="dx")
    plan.set_defaults(handler=command_plan)
    arguments = parser.parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
