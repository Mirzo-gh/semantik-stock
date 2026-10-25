#!/usr/bin/env python3
"""Alimente le stock de jours pré-générés (R2 privé semantik-private/daily-stock/) depuis GitHub Actions.

Ce fichier vit dans un dépôt PUBLIC : son journal l'est aussi. Règle absolue : aucun mot, aucun contenu de fichier, aucune empreinte, aucun chemin
de donnée dans ce qui est affiché. On n'affiche que des étapes et des compteurs. Les sorties des outils (pipeline, boto3) vont dans un fichier
temporaire qui n'est JAMAIS relu ni affiché ; un échec n'affiche que l'étape et le code de sortie.

Données lues dans R2 privé (jamais commitées) : pipeline-data/pipeline-files.tar.gz (candidats, filtres, difficultés…), pipeline-data/data-work.tar.gz
(vocabulaire + vecteurs construits par le déploiement local), fasttext/ft.npz, exclusions/exclusions.tsv. Le vocabulaire n'est JAMAIS écrit ici :
si celui de la prod n'est plus celui des données, on s'arrête (il faut republier les données depuis le Mac : scripts/local-deploy.sh --publish-data).
Env : SEMANTIK_MASTER_KEY, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY ; STOCK_DAYS_AHEAD (défaut 90) ; STOCK_DRY_RUN=1 (rien n'est écrit).
DEPLOY_CHECK_TOKEN (facultatif ; sans lui, J+1 et J+2 sont repris du stock R2 avec un avertissement et arrêt s'ils y manquent, sans comparaison à la prod) : jeton de lecture des jours futurs en ligne (en-tête X-Deploy-Check, HMAC du secret de session, comme le Deploy). J+1 et J+2 sont
lus avec lui et ÉPINGLÉS ; sans jeton, ou si la lecture échoue, on s'arrête : jamais de tirage à frais pour un jour déjà publié.
STOCK_CHANGE_NUMBER (workflow change.yml) : numéro d'un jour futur (>= J+3) dont l'admin a demandé le changement de mot. La demande est lue dans R2 privé
(admin/requests/<n>.json, jamais dans une entrée du workflow) ; le résultat (codes et booléens seulement, aucun mot) est écrit dans admin/results/<n>.json. Les mots
imposés vivent dans admin/overrides.tsv (numéro<TAB>mot, PRIVÉ), relu à chaque run et par le Deploy local.
Essais hors ligne : STOCK_LOCAL_STORAGE=<dossier miroir du bucket>, STOCK_LOCAL_SITE=<dossier miroir de /data/daily du site> ;
STOCK_LOCAL_SITE_MASKED=<numéros séparés par des virgules> : ces jours du miroir ne sont servis qu'avec le bon jeton (STOCK_LOCAL_SITE_TOKEN).
"""
import array
import base64
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zoneinfo

SITE = "https://seman-tik.fr"
BUCKET = "semantik-private"
STOCK_PREFIX = "daily-stock/"
HERE = pathlib.Path(__file__).resolve().parent
SILENT_LOG = pathlib.Path(tempfile.mkdtemp(prefix="stock-silent-")) / "tools.log"  # sorties d'outils : jamais affichées


class Stop(Exception):
    """Arrêt contrôlé : le message est PUBLIC, il ne contient jamais de donnée."""


REBUILD_CONTEXT: dict = {}


class Refused(Exception):
    """Demande de changement de mot refusée (résultat déjà écrit dans R2) : le run sort en échec pour qu'un essai refusé ne compte jamais comme un essai réussi."""


def log(message: str) -> None:
    print(message, flush=True)


def run_silent(label: str, command: list, cwd=None, env=None) -> None:
    with open(SILENT_LOG, "ab") as sink:
        code = subprocess.run(command, cwd=cwd, env=env, stdout=sink, stderr=sink, check=False).returncode
    if code != 0:
        raise Stop(f"échec de l'étape « {label} » (code {code})")


class Storage:
    """R2 privé via l'API S3 (jeton limité au bucket), ou un dossier local pour les essais."""

    def __init__(self):
        self.local = os.environ.get("STOCK_LOCAL_STORAGE")
        if self.local:
            return
        import boto3  # importé ici : inutile hors ligne

        account = os.environ["R2_ACCOUNT_ID"]
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{account}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )

    def get(self, key: str, destination: pathlib.Path) -> bool:
        """True si l'objet existe (copié vers destination), False si absent ; toute autre erreur arrête."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        if self.local:
            source = pathlib.Path(self.local) / key
            if not source.exists():
                return False
            shutil.copyfile(source, destination)
            return True
        try:
            self.client.download_file(BUCKET, key, str(destination))
            return True
        except Exception as error:  # noqa: BLE001 — seul le code de statut est examiné, jamais affiché
            code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise Stop(f"lecture R2 impossible ({type(error).__name__})") from None

    def delete(self, key: str) -> None:
        if self.local:
            (pathlib.Path(self.local) / key).unlink(missing_ok=True)
            return
        try:
            self.client.delete_object(Bucket=BUCKET, Key=key)
        except Exception as error:  # noqa: BLE001
            raise Stop(f"suppression R2 impossible ({type(error).__name__})") from None

    def put(self, key: str, source: pathlib.Path) -> None:
        if self.local:
            target = pathlib.Path(self.local) / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return
        try:
            self.client.upload_file(str(source), BUCKET, key, ExtraArgs={"ContentType": "application/json"})
        except Exception as error:  # noqa: BLE001
            raise Stop(f"écriture R2 impossible ({type(error).__name__})") from None


def site_get(path: str, destination: pathlib.Path, token: str = "") -> str:
    """Fichier du site : 'present' (copié) / 'absent' (404 ou page HTML) / 'erreur'. Avec `token`, l'en-tête X-Deploy-Check autorise les jours futurs."""
    local_site = os.environ.get("STOCK_LOCAL_SITE")
    if local_site:
        source = pathlib.Path(local_site) / path
        masked = {item for item in os.environ.get("STOCK_LOCAL_SITE_MASKED", "").split(",") if item}
        if path.startswith("daily/") and path[len("daily/"):-len(".json")] in masked and (not token or token != os.environ.get("STOCK_LOCAL_SITE_TOKEN")):
            return "absent"  # masqué au public jusqu'à minuit Paris
        if not source.exists():
            return "absent"
        shutil.copyfile(source, destination)
        return "present"
    headers = {"User-Agent": "semantik-stock"}
    if token:
        headers["X-Deploy-Check"] = token
    request = urllib.request.Request(f"{SITE}/data/{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if "text/html" in response.headers.get("content-type", ""):
                return "absent"
            destination.write_bytes(response.read())
            return "present"
    except urllib.error.HTTPError as error:
        return "absent" if error.code == 404 else "erreur"
    except Exception:  # noqa: BLE001
        return "erreur"


def safe_extract(archive: pathlib.Path, destination: pathlib.Path) -> None:
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():
                raise Stop("archive de données invalide")
        bundle.extractall(destination, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def read_launch_date() -> datetime.date:
    return datetime.date.fromisoformat((HERE / "launch.txt").read_text().strip())


def daily_arguments(files: pathlib.Path, models: pathlib.Path) -> list:
    """Arguments facultatifs de `cli daily`, dans le même ordre et sous les mêmes conditions que .github/workflows/deploy.yml (parité testée)."""
    arguments: list = []
    blend_from = files / "blend_from.txt"
    if blend_from.exists():
        arguments += ["--blend-from", blend_from.read_text().strip(), "--fasttext", str(models / "ft.npz"), "--jdm", str(files / "jdm_associations.tsv"), "--filter", str(files / "filter_lemmas.tsv")]
        if (models / "exclusions.tsv").exists():
            arguments += ["--exclusions", str(models / "exclusions.tsv")]
    if (files / "v3_from.txt").exists():
        arguments += ["--v3-from", (files / "v3_from.txt").read_text().strip()]
    if (files / "difficulty.tsv").exists():
        arguments += ["--difficulty", str(files / "difficulty.tsv")]
    if (files / "cleanup_from.txt").exists():
        arguments += ["--cleanup-from", (files / "cleanup_from.txt").read_text().strip(), "--names", str(files / "filter_names.tsv")]
    if (files / "start_from.txt").exists():
        arguments += ["--start-from", (files / "start_from.txt").read_text().strip(), "--start-exclude", str(files / "vocab_blocklist.txt"), str(files / "filter_lemmas.tsv"), str(files / "filter_names.tsv")]
    pos_file = files / "pos_lemmas.tsv"
    if pos_file.exists() and pos_file.stat().st_size > 0:
        arguments += ["--pos", str(pos_file)]
    return arguments


def pipeline_command(data: pathlib.Path, date: datetime.date, number: int, candidates: pathlib.Path, work: pathlib.Path, files: pathlib.Path, models: pathlib.Path, overrides: pathlib.Path | None = None) -> list:
    command = [sys.executable, "-m", "semantik_pipeline.cli", "daily", "--data", str(data), "--date", date.isoformat(), "--number", str(number), "--candidates", str(candidates), "--pinned", str(work / "pinned")]
    if overrides is not None and overrides.exists() and overrides.stat().st_size > 0:
        command += ["--force-word-file", str(overrides)]  # mots imposés par l'admin (PRIVÉ) ; les jours déjà générés ne sont pas touchés
    return command + daily_arguments(files, models)


OVERRIDES_KEY = "admin/overrides.tsv"
CHANGE_CONTEXT: dict = {}  # numéro/mode/stockage de la demande en cours : sert à écrire un résultat « error » si le run s'arrête


def read_overrides(path: pathlib.Path) -> dict:
    table: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            number, _, word = line.partition("\t")
            if number.strip().isdigit() and word.strip():
                table[int(number)] = word.strip()
    return table


def write_overrides(path: pathlib.Path, table: dict) -> None:
    path.write_text("".join(f"{number}\t{word}\n" for number, word in sorted(table.items())), encoding="utf-8")


def write_result(storage: "Storage", work: pathlib.Path, number: int, mode: str, status: str, codes: list, unreviewed, stock_last: int, dry_run: bool) -> None:
    """admin/results/<n>.json : statut, codes de refus et booléens seulement. Jamais un mot, une taille de mot ni un indice dérivé (le journal et ce fichier sont lus au-delà du secret)."""
    result = {"v": 1, "number": number, "mode": mode, "status": status, "codes": codes, "neighbours_unreviewed": unreviewed, "dry_run": dry_run,
              "stock_last": stock_last, "at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    path = work / f"result-{number}.json"
    path.write_text(json.dumps(result))
    storage.put(f"admin/results/{number}.json", path)
    log(f"Résultat du changement n°{number} : {status}" + (f" ({', '.join(codes)})" if codes else ""))


def export_pool(storage: "Storage", work: pathlib.Path, data: pathlib.Path, files: pathlib.Path, candidates: pathlib.Path, environment: dict, last: int) -> None:
    """admin/pool.json (R2 privé, lu par l'admin) : candidats de tirage et mots en clair des jours 1..last. Écrit seulement dans un run réel, jamais journalisé ni commité."""
    pool_path = work / "pool.json"
    command = [sys.executable, "-m", "semantik_pipeline.cli", "pool", "--data", str(data), "--last", str(last), "--candidates", str(candidates), "--out", str(pool_path)]
    if (files / "difficulty.tsv").exists():
        command += ["--difficulty", str(files / "difficulty.tsv")]
    run_silent("export du pool admin", command, cwd=work, env=environment)
    storage.put("admin/pool.json", pool_path)
    log("Pool admin exporté.")


def check_start_regeneration(work, data, daily_dir, files, models, candidates, environment, launch, today_number, stock_first, stock_last) -> None:
    """Essai à blanc de l'activation du mot de départ : regénère (dans un dossier à part) les jours du stock qui seraient supprimés, puis exige
    « mêmes mots, mêmes scores, plus start ». Ne touche ni R2 ni le stock ; le journal ne donne que des compteurs."""
    start_from = int((files / "start_from.txt").read_text().strip())
    first = max(start_from, today_number + 3, stock_first)  # le jour courant, J+1 et J+2 sont en ligne : jamais refaits
    if first > stock_last:
        log("Mot de départ : aucun jour du stock à comparer (stock plus court que le premier jour concerné)")
        return
    sys.path.insert(0, str(HERE / "pipeline"))
    from semantik_pipeline import start_check  # numpy requis, comme le pipeline

    simulated = work / "data-start-check"
    shutil.rmtree(simulated, ignore_errors=True)
    simulated.mkdir()
    for item in data.iterdir():
        if item.name != "daily":
            (simulated / item.name).symlink_to(item.resolve())
    (simulated / "daily").mkdir()
    for path in daily_dir.glob("*.json"):
        if int(path.stem) < first:
            shutil.copyfile(path, simulated / "daily" / path.name)
    date = launch + datetime.timedelta(days=stock_last - 1)
    run_silent("essai à blanc du mot de départ", pipeline_command(simulated, date, stock_last, candidates, work, files, models), cwd=work, env=environment)
    subsets = {}
    for label, source in (("avant", daily_dir), ("apres", simulated / "daily")):  # seuls les jours qui seraient refaits sont comparés
        subsets[label] = work / f"start-check-{label}"
        shutil.rmtree(subsets[label], ignore_errors=True)
        subsets[label].mkdir()
        for path in source.glob("*.json"):
            if int(path.stem) >= first:
                shutil.copyfile(path, subsets[label] / path.name)
    counts, problems = start_check.compare_days(subsets["avant"], subsets["apres"], first)
    compared_range = f"n°{first} à {stock_last}"
    if problems:
        for problem in problems[:10]:
            log(problem)
        raise Stop(f"mot de départ : {len(problems)} écart(s) entre le stock existant et la régénération à blanc ({compared_range}) : ne rien supprimer")
    log(f"Mot de départ : régénération à blanc identique au stock plus `start` ({counts['compared']} jours comparés, {counts['with_start']} avec start, {compared_range})")


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SCORE_TOLERANCE = 1  # unité stockée = 0,01 °C : un jour généré sur Mac et le même sur Linux diffèrent d'au plus 1 sur certains scores


def same_published_day(stock_copy: pathlib.Path, published: pathlib.Path) -> bool:
    """Même jour publié ? Octet pour octet, sinon : mêmes champs hors scores/start, même `top` (mêmes mots, même ordre) et scores à ±SCORE_TOLERANCE.
    `start` est un indice dérivé des scores : il peut varier de ±1 sans changer de mot du jour, donc il n'entre pas dans la comparaison."""
    if sha(stock_copy) == sha(published):
        return True
    try:
        stock_day, published_day = json.loads(stock_copy.read_text()), json.loads(published.read_text())
        if not isinstance(stock_day, dict) or not isinstance(published_day, dict):
            return False
        ignored = {"scores", "start"}
        if {key: value for key, value in stock_day.items() if key not in ignored} != {key: value for key, value in published_day.items() if key not in ignored}:
            return False  # inclut `top` (même graine, donc mêmes octets masqués = mêmes mots dans le même ordre), seed, vocab, forms, pos
        spec = importlib.util.spec_from_file_location("stock_masking", HERE / "pipeline" / "semantik_pipeline" / "masking.py")
        masking = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(masking)
        seed = published_day["seed"]
        decoded = []
        for day in (stock_day, published_day):
            values = array.array("H")
            values.frombytes(masking.xor_mask(base64.b64decode(day["scores"]), seed))
            decoded.append(values)
        if len(decoded[0]) != len(decoded[1]):
            return False
        return all(abs(left - right) <= SCORE_TOLERANCE for left, right in zip(*decoded))
    except (ValueError, KeyError, TypeError):
        return False


def prepare_change(number: int, storage: "Storage", work: pathlib.Path, data: pathlib.Path, daily_dir: pathlib.Path, files: pathlib.Path, models: pathlib.Path,
                   candidates: pathlib.Path, environment: dict, overrides: pathlib.Path, existing: dict, today_number: int, stock_first: int, stock_last: int, dry_run: bool):
    """Demande de changement de mot (jour futur >= J+3). Renvoie le nouveau dernier jour du stock (purgé à partir de N) pour que la génération continue,
    ou None si simple aperçu (résultat écrit) ; une demande refusée lève Refused (résultat écrit). Rien n'est affiché de ce qui touche au mot : codes et booléens seulement."""
    request_path = work / "request.json"
    if not storage.get(f"admin/requests/{number}.json", request_path):
        write_result(storage, work, number, "?", "refused", ["no_request"], None, stock_last, dry_run)
        raise Refused()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        mode = request.get("mode") if isinstance(request, dict) else None
    except ValueError:
        mode = None
    if mode not in ("word", "redraw", "revert"):
        write_result(storage, work, number, "?", "refused", ["bad_request"], None, stock_last, dry_run)
        raise Refused()
    CHANGE_CONTEXT.update(number=number, mode=mode, storage=storage, work=work, stock_last=stock_last, dry_run=dry_run)
    if number <= today_number + 2:
        write_result(storage, work, number, mode, "refused", ["protected"], None, stock_last, dry_run)  # J..J+2 : hors bouton (Deploy local)
        raise Refused()
    if number > stock_last or number < stock_first:
        write_result(storage, work, number, mode, "refused", ["out_of_stock"], None, stock_last, dry_run)
        raise Refused()
    resolution_path = work / "resolution.json"
    command = [sys.executable, "-m", "semantik_pipeline.cli", "change", "--data", str(data), "--number", str(number), "--request", str(request_path),
               "--candidates", str(candidates), "--out", str(resolution_path)]
    if (files / "difficulty.tsv").exists():
        command += ["--difficulty", str(files / "difficulty.tsv")]
    if (models / "exclusions.tsv").exists():
        command += ["--exclusions", str(models / "exclusions.tsv")]
    run_silent("contrôle du changement de mot", command, cwd=work, env=environment)
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    if resolution["codes"]:
        write_result(storage, work, number, mode, "refused", resolution["codes"], None, stock_last, dry_run)
        raise Refused()
    unreviewed = resolution["neighbours_unreviewed"]
    CHANGE_CONTEXT["unreviewed"] = unreviewed
    if dry_run:
        write_result(storage, work, number, mode, "ok", [], unreviewed, stock_last, True)  # aperçu : mot accepté, rien n'est modifié
        log("ESSAI : rien n'a été modifié dans R2.")
        return None

    # Changement réel. 1. sauvegarde de l'ancien fichier du jour, 2. mot imposé inscrit (avant toute purge : une reprise le retrouve), 3. purge du stock >= N
    old_file = daily_dir / f"{number}.json"
    storage.put(f"admin/backup/{number}-{sha(old_file)[:8]}.json", old_file)
    table = read_overrides(overrides)
    if mode == "revert":
        table.pop(number, None)
    else:
        table[number] = resolution["word"]
    write_overrides(overrides, table)
    storage.put(OVERRIDES_KEY, overrides)
    kept = work / "manifest-purge.json"
    kept.write_text(json.dumps({"v": 1, "first": stock_first, "last": number - 1, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    storage.put(STOCK_PREFIX + "manifest.json", kept)
    for purged in range(number, stock_last + 1):
        storage.delete(f"{STOCK_PREFIX}{purged}.json")
        (daily_dir / f"{purged}.json").unlink(missing_ok=True)
        existing.pop(f"{purged}.json", None)
    log(f"Stock purgé à partir du n°{number} ({stock_last - number + 1} jours) ; régénération sur la nouvelle chaîne")
    return number - 1


def prepare_rebuild(first_number: int, storage: "Storage", work: pathlib.Path, daily_dir: pathlib.Path, existing: dict, today_number: int, stock_first: int, stock_last: int, dry_run: bool) -> int:
    """Reconstruction du stock à partir du jour `first_number` (liste de candidats ou filtre de tirage modifié : toute la chaîne change à partir de là).
    Les jours >= first_number sont retirés du dossier local (et, en réel seulement, de R2) puis régénérés par le tirage normal ; les jours avant restent identiques.
    Journal : des nombres seulement (combien de jours changent de mot), jamais un mot. Renvoie le nouveau dernier jour du stock."""
    if first_number <= today_number:
        raise Stop("reconstruction : le jour courant et les jours passés sont protégés")
    if first_number > stock_last:
        raise Stop("reconstruction : numéro au-delà de la fin du stock")
    kept = work / "rebuild-before"
    kept.mkdir()
    for number in range(first_number, stock_last + 1):
        source = daily_dir / f"{number}.json"
        if source.exists():
            shutil.copyfile(source, kept / source.name)
    log(f"Reconstruction à partir du n°{first_number} : {stock_last - first_number + 1} jours du stock à refaire ({'ESSAI' if dry_run else 'RÉEL'}).")
    if not dry_run:
        manifest = work / "manifest-rebuild.json"
        manifest.write_text(json.dumps({"v": 1, "first": stock_first, "last": first_number - 1, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat()}))
        storage.put(STOCK_PREFIX + "manifest.json", manifest)
        for purged in range(first_number, stock_last + 1):
            storage.delete(f"{STOCK_PREFIX}{purged}.json")
    for purged in range(first_number, stock_last + 1):
        (daily_dir / f"{purged}.json").unlink(missing_ok=True)
        existing.pop(f"{purged}.json", None)
    REBUILD_CONTEXT["before"] = kept
    REBUILD_CONTEXT["first"] = first_number
    return first_number - 1


def target_index(path: pathlib.Path):
    """Index du mot du jour d'un fichier daily (jamais affiché), None si illisible : on compare les MOTS, pas les scores (écarts Mac/Linux de ±1 ou plus sur quelques scores)."""
    try:
        sys.path.insert(0, str(HERE / "pipeline"))
        from semantik_pipeline import daily as daily_module
        return daily_module.decode_target_index(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001
        return None


def report_rebuild(daily_dir: pathlib.Path, last: int) -> None:
    """Nombres seulement : combien de jours reconstruits ont changé de mot par rapport au stock d'avant."""
    kept, first = REBUILD_CONTEXT["before"], REBUILD_CONTEXT["first"]
    compared = [number for number in range(first, last + 1) if (kept / f"{number}.json").exists() and (daily_dir / f"{number}.json").exists()]
    before_after = {number: (target_index(kept / f"{number}.json"), target_index(daily_dir / f"{number}.json")) for number in compared}
    changed = [number for number, (before, after) in before_after.items() if before is None or after is None or before != after]  # illisible = compté comme changé
    log(f"Reconstruction : {len(changed)} jours sur {len(compared)} ont changé de mot (n°{first} à n°{last}).")


def finish_change(number: int, storage: "Storage", work: pathlib.Path, stock_last: int) -> None:
    """Fin d'un changement réel : résultat, audit sans mot, demande supprimée (seulement après succès)."""
    mode = CHANGE_CONTEXT.get("mode", "?")
    write_result(storage, work, number, mode, "ok", [], CHANGE_CONTEXT.get("unreviewed"), stock_last, False)
    audit = work / "audit.json"
    audit.write_text(json.dumps({"v": 1, "number": number, "mode": mode, "at": datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    storage.put(f"admin/audit/{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}-{number}.json", audit)
    storage.delete(f"admin/requests/{number}.json")


def report_failure() -> None:
    """Un changement s'est arrêté après avoir commencé : on le dit dans le résultat (la demande reste, le prochain run reprend)."""
    if not CHANGE_CONTEXT or CHANGE_CONTEXT.get("dry_run"):
        return
    try:
        write_result(CHANGE_CONTEXT["storage"], CHANGE_CONTEXT["work"], CHANGE_CONTEXT["number"], CHANGE_CONTEXT["mode"], "error", ["stopped"], None, CHANGE_CONTEXT["stock_last"], False)
    except Exception:  # noqa: BLE001
        pass



def vocab_extends(old_lines: list[str], new_lines: list[str]) -> bool:
    """Vrai si new_lines prolonge old_lines : mêmes index, un lemme peut être neutralisé (~) mais jamais déplacé ni remplacé."""
    return len(new_lines) >= len(old_lines) and all(new == old or new == "~" + old or old == "~" + new for old, new in zip(old_lines, new_lines))


def main() -> None:
    dry_run = os.environ.get("STOCK_DRY_RUN") == "1"
    days_ahead = int(os.environ.get("STOCK_DAYS_AHEAD", "90"))
    master_key = os.environ.get("SEMANTIK_MASTER_KEY", "")
    if not master_key:
        raise Stop("clé maître absente")
    work = pathlib.Path(os.environ.get("STOCK_WORK", "work")).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    deploy_token = os.environ.get("DEPLOY_CHECK_TOKEN", "")  # facultatif : sans lui, J+1 et J+2 sont repris du stock (voir plus bas)
    storage = Storage()
    log(f"Mode : {'ESSAI (rien n’est écrit)' if dry_run else 'RÉEL'} ; avance visée : {days_ahead} jours")

    # 1. Données privées
    for key, name in (("pipeline-data/pipeline-files.tar.gz", "fichiers du pipeline"), ("pipeline-data/data-work.tar.gz", "vocabulaire et vecteurs")):
        if not storage.get(key, work / pathlib.Path(key).name):
            raise Stop(f"{name} absents de R2 (publier avec scripts/local-deploy.sh --publish-data)")
    files = work / "pipeline-files"
    safe_extract(work / "pipeline-files.tar.gz", files)
    safe_extract(work / "data-work.tar.gz", work)
    data = work / "data-work"
    models = work / "models"
    models.mkdir()
    bundle_info = json.loads((data / "BUNDLE.json").read_text())
    if (files / "blend_from.txt").exists():
        if not storage.get("fasttext/ft.npz", models / "ft.npz"):
            raise Stop("vecteurs fastText absents de R2")
        if bundle_info.get("ft_sha256") and sha(models / "ft.npz") != bundle_info["ft_sha256"]:
            raise Stop("vecteurs fastText différents de ceux des données publiées")
        storage.get("exclusions/exclusions.tsv", models / "exclusions.tsv")
    overrides = work / "overrides.tsv"
    storage.get(OVERRIDES_KEY, overrides)  # mots imposés par l'admin : absent = aucun
    log(f"Données privées chargées ; mots imposés : {len(read_overrides(overrides))}")

    # 2. Le vocabulaire des données doit être celui de la prod (lecture seule, jamais réécrit)
    vocab_hash = bundle_info["lemmas"]
    probe = work / "vocab-probe"
    if site_get(f"lemmas.{vocab_hash}.txt", probe) != "present":
        raise Stop("le vocabulaire des données n'est plus en ligne : republier les données (scripts/local-deploy.sh --publish-data) après le déploiement")
    probe.unlink(missing_ok=True)
    log("Vocabulaire des données = vocabulaire en ligne")

    # 3. Numéros de jour (Paris)
    launch = read_launch_date()
    today_paris = datetime.datetime.now(zoneinfo.ZoneInfo("Europe/Paris")).date()
    today_number = (today_paris - launch).days + 1
    target_number = today_number + days_ahead
    target_date = launch + datetime.timedelta(days=target_number - 1)
    log(f"Jour courant n°{today_number}, génération jusqu'au n°{target_number}")

    # 4. Historique : jours passés EN LIGNE (publics), stock déjà déposé (jours >= courant)
    daily_dir = data / "daily"
    shutil.rmtree(daily_dir, ignore_errors=True)
    daily_dir.mkdir()
    for number in range(1, today_number):
        if site_get(f"daily/{number}.json", daily_dir / f"{number}.json") != "present":
            raise Stop(f"jour passé n°{number} introuvable en ligne : historique incomplet, arrêt")
    if site_get(f"daily/{today_number}.json", daily_dir / f"{today_number}.json") != "present":
        raise Stop("jour courant introuvable ou illisible en ligne : impossible de vérifier le vocabulaire courant")
    # Le vocabulaire des données doit être le vocabulaire COURANT (celui du jour en ligne), pas seulement un ancien fichier encore hébergé
    current_vocab = json.loads((daily_dir / f"{today_number}.json").read_text()).get("vocab")
    if current_vocab != vocab_hash:
        # Après un changement de vocabulaire, les jours J..J+2 restent gelés à l'ancien (freeze_days.py) : le jour courant a l'ancien hash tant que le nouveau n'est pas son tour.
        # Les données sont acceptées si leur vocabulaire PROLONGE exactement celui du jour courant (append-only, mêmes règles que check_vocab_append_only.py).
        current_file = work / "vocab-current"
        if not current_vocab or site_get(f"lemmas.{current_vocab}.txt", current_file) != "present" or not vocab_extends(
            current_file.read_text(encoding="utf-8").splitlines(), (data / f"lemmas.{vocab_hash}.txt").read_text(encoding="utf-8").splitlines()
        ):
            raise Stop("le vocabulaire des données n'est pas le vocabulaire courant : republier les données (scripts/local-deploy.sh --publish-data)")
        log("Vocabulaire des données = prolongement du vocabulaire du jour courant (gel en cours)")
    manifest_path = work / "manifest.json"
    stock_first, stock_last = today_number, today_number - 1
    if storage.get(STOCK_PREFIX + "manifest.json", manifest_path):
        manifest = json.loads(manifest_path.read_text())
        stock_first, stock_last = manifest["first"], manifest["last"]
    existing = {}
    stock_copies = work / "stock-copies"  # copies du stock pour J..J+2 : comparées à la prod, jamais utilisées pour le tirage
    stock_copies.mkdir()
    for number in range(max(stock_first, today_number), stock_last + 1):
        destination = stock_copies / f"{number}.json" if number <= today_number + 2 else daily_dir / f"{number}.json"
        if not storage.get(f"{STOCK_PREFIX}{number}.json", destination):
            if number <= today_number + 2:
                continue  # J..J+2 : la prod fait foi, le stock n'a pas besoin de les avoir
            raise Stop(f"jour n°{number} annoncé par le manifeste mais absent du stock")
    # J+1 et J+2 sont déjà EN LIGNE (publiés d'avance, masqués au public jusqu'à minuit Paris) : on les lit avec le jeton du Deploy et on les ÉPINGLE, même si
    # le stock en avait une autre version. Pas de jeton ou lecture impossible : arrêt, jamais de tirage à frais pour un jour déjà publié (cela décalerait tout).
    for number in (today_number + 1, today_number + 2):
        destination = daily_dir / f"{number}.json"
        if deploy_token:
            if site_get(f"daily/{number}.json", destination, deploy_token) != "present":
                raise Stop(f"jour n°{number} (J+{number - today_number}) illisible en ligne avec le jeton : arrêt, aucun tirage à frais pour un jour déjà publié")
        else:
            # Sans jeton (secret de session indisponible) : le stock fait foi pour J+1 et J+2, sans comparaison possible à la prod ; s'il y manque un jour, arrêt
            stock_copy = stock_copies / f"{number}.json"
            if not stock_copy.exists():
                raise Stop(f"jeton absent et jour n°{number} (J+{number - today_number}) absent du stock : arrêt, aucun tirage à frais pour un jour déjà publié")
            shutil.copyfile(stock_copy, destination)
            log(f"AVERTISSEMENT : jeton absent, jour n°{number} repris du stock sans comparaison à la prod")
        try:
            pinned_day = json.loads(destination.read_text())
        except ValueError:
            raise Stop(f"jour n°{number} illisible (JSON invalide) : arrêt") from None
        if pinned_day.get("n") != number or pinned_day.get("vocab") not in (vocab_hash, current_vocab):  # J..J+2 peuvent rester gelés à l'ancien vocabulaire
            raise Stop(f"jour n°{number} : numéro ou vocabulaire inattendu : arrêt")
    # Garde-fou : un jour J..J+2 déjà dans le stock doit être le MÊME que la prod (mêmes mots, même ordre, scores à ±1 près : Mac/Linux). Sinon le stock est sur une autre chaîne de mots
    # (le tirage dépend de l'historique) : arrêt, rien n'est généré ni déposé ; refaire le stock à partir des jours publiés.
    differing = [path.stem for path in sorted(stock_copies.glob("*.json")) if not same_published_day(path, daily_dir / path.name)]
    if differing:
        raise Stop(f"{len(differing)} jour(s) du stock parmi J..J+2 (n°{', n°'.join(differing)}) diffèrent de la prod : arrêt, le stock n'est pas sur la chaîne publiée")
    for path in daily_dir.glob("*.json"):
        existing[path.name] = sha(path)
    log(f"Historique chargé : {today_number - 1} jours passés, stock existant jusqu'au n°{stock_last}")
    candidates = work / "candidates.txt"
    candidates.write_text("\n".join(line.split("\t")[1] for line in (files / "candidates.tsv").read_text(encoding="utf-8").splitlines()[1:]) + "\n", encoding="utf-8")
    (work / "pinned").mkdir()
    environment = dict(os.environ, SEMANTIK_MASTER_KEY=master_key, PYTHONPATH=str(HERE / "pipeline"))
    if dry_run and (files / "start_from.txt").exists() and not os.environ.get("STOCK_REBUILD_FROM"):  # pendant une reconstruction tout change par construction
        # Mot de départ : AVANT toute suppression dans R2, on regénère à blanc les jours du stock concernés et on les compare à l'existant
        check_start_regeneration(work, data, daily_dir, files, models, candidates, environment, launch, today_number, stock_first, stock_last)
    change_number = os.environ.get("STOCK_CHANGE_NUMBER", "").strip()
    old_stock_last = stock_last
    if os.environ.get("STOCK_CHANGE_REQUIRED") == "1" and not change_number:
        raise Stop("numéro de jour à changer absent")
    if change_number:
        if not change_number.isdigit():
            raise Stop("numéro de jour à changer invalide")
        outcome = prepare_change(int(change_number), storage, work, data, daily_dir, files, models, candidates, environment, overrides, existing, today_number, stock_first, stock_last, dry_run)
        if outcome is None:
            return  # refusé ou aperçu : le résultat est écrit, rien d'autre n'a changé
        stock_last = outcome
        target_number = max(target_number, old_stock_last)
    rebuild_from = os.environ.get("STOCK_REBUILD_FROM", "").strip()
    if os.environ.get("STOCK_REBUILD_REQUIRED") == "1" and not rebuild_from:
        raise Stop("numéro de reconstruction absent")
    if rebuild_from:
        if not rebuild_from.isdigit() or change_number:
            raise Stop("numéro de reconstruction invalide")
        stock_last = prepare_rebuild(int(rebuild_from), storage, work, daily_dir, existing, today_number, stock_first, stock_last, dry_run)
        target_number = max(target_number, old_stock_last)
    if stock_last >= target_number:
        log("Stock déjà suffisant : rien à générer")
        if not dry_run:
            export_pool(storage, work, data, files, candidates, environment, stock_last)
        return

    # 5. Génération (les jours existants sont conservés tels quels par le pipeline)
    target_date = launch + datetime.timedelta(days=target_number - 1)
    command = pipeline_command(data, target_date, target_number, candidates, work, files, models, overrides)
    run_silent("génération des jours", command, cwd=work, env=environment)

    if rebuild_from:
        report_rebuild(daily_dir, target_number)

    # 6. Contrôles : rien d'existant n'a changé, tout est là, numérotation cohérente
    for name, digest in existing.items():
        if sha(daily_dir / name) != digest:
            raise Stop(f"un jour existant a changé (n°{name[:-5]}) : arrêt")
    new_numbers = []
    for number in range(today_number, target_number + 1):
        path = daily_dir / f"{number}.json"
        if not path.exists() or path.stat().st_size == 0:
            raise Stop(f"jour n°{number} non généré")
        if json.loads(path.read_text())["n"] != number:
            raise Stop(f"jour n°{number} : numéro interne incohérent")
        if number > stock_last or number < stock_first:
            new_numbers.append(number)
    # Le jour courant déjà en ligne n'est pas redéposé dans le stock (les actifs priment) ; les autres jours manquants le sont
    new_numbers = [number for number in new_numbers if f"{number}.json" not in existing]
    log(f"Jours prêts : {target_number - today_number + 1} ; nouveaux à déposer : {len(new_numbers)}")

    if dry_run:
        log("ESSAI : rien n'a été déposé dans R2.")
        return

    # 7. Dépôt (jours d'abord, manifeste en dernier)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda number: storage.put(f"{STOCK_PREFIX}{number}.json", daily_dir / f"{number}.json"), new_numbers))
    new_manifest = {"v": 1, "first": min(stock_first, today_number) if stock_last >= stock_first else today_number, "last": target_number, "updated": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    manifest_out = work / "manifest-new.json"
    manifest_out.write_text(json.dumps(new_manifest))
    storage.put(STOCK_PREFIX + "manifest.json", manifest_out)
    # Relecture : le dernier jour déposé est identique au fichier généré
    check = work / "last-check.json"
    if not storage.get(f"{STOCK_PREFIX}{target_number}.json", check) or sha(check) != sha(daily_dir / f"{target_number}.json"):
        raise Stop("vérification du dernier jour déposé en échec")
    log(f"Stock à jour : jours n°{today_number} à {target_number} ({len(new_numbers)} déposés).")
    export_pool(storage, work, data, files, candidates, environment, target_number)
    if change_number:
        finish_change(int(change_number), storage, work, target_number)


if __name__ == "__main__":
    try:
        main()
    except Refused:
        log("ARRÊT : changement refusé (voir le résultat)")
        sys.exit(1)
    except Stop as stop:
        log(f"ARRÊT : {stop}")
        report_failure()
        sys.exit(1)
    except Exception as error:  # noqa: BLE001 — le type seul, jamais le message ni la trace (ils pourraient contenir une donnée)
        log(f"ERREUR INATTENDUE : {type(error).__name__}")
        sys.exit(1)
