#!/usr/bin/env python3
"""Alimente le stock de jours pré-générés (R2 privé semantik-private/daily-stock/) depuis GitHub Actions.

Ce fichier vit dans un dépôt PUBLIC : son journal l'est aussi. Règle absolue : aucun mot, aucun contenu de fichier, aucune empreinte, aucun chemin
de donnée dans ce qui est affiché. On n'affiche que des étapes et des compteurs. Les sorties des outils (pipeline, boto3) vont dans un fichier
temporaire qui n'est JAMAIS relu ni affiché ; un échec n'affiche que l'étape et le code de sortie.

Données lues dans R2 privé (jamais commitées) : pipeline-data/pipeline-files.tar.gz (candidats, filtres, difficultés…), pipeline-data/data-work.tar.gz
(vocabulaire + vecteurs construits par le déploiement local), fasttext/ft.npz, exclusions/exclusions.tsv. Le vocabulaire n'est JAMAIS écrit ici :
si celui de la prod n'est plus celui des données, on s'arrête (il faut republier les données depuis le Mac : scripts/local-deploy.sh --publish-data).
Env : SEMANTIK_MASTER_KEY, R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY ; STOCK_DAYS_AHEAD (défaut 90) ; STOCK_DRY_RUN=1 (rien n'est écrit).
Essais hors ligne : STOCK_LOCAL_STORAGE=<dossier miroir du bucket>, STOCK_LOCAL_SITE=<dossier miroir de /data/daily du site>.
"""
import concurrent.futures
import datetime
import hashlib
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


def site_get(path: str, destination: pathlib.Path) -> str:
    """Fichier PUBLIC du site : 'present' (copié) / 'absent' (404 ou page HTML) / 'erreur'."""
    local_site = os.environ.get("STOCK_LOCAL_SITE")
    if local_site:
        source = pathlib.Path(local_site) / path
        if not source.exists():
            return "absent"
        shutil.copyfile(source, destination)
        return "present"
    request = urllib.request.Request(f"{SITE}/data/{path}", headers={"User-Agent": "semantik-stock"})
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


def pipeline_command(data: pathlib.Path, date: datetime.date, number: int, candidates: pathlib.Path, work: pathlib.Path, files: pathlib.Path, models: pathlib.Path) -> list:
    command = [sys.executable, "-m", "semantik_pipeline.cli", "daily", "--data", str(data), "--date", date.isoformat(), "--number", str(number), "--candidates", str(candidates), "--pinned", str(work / "pinned")]
    return command + daily_arguments(files, models)


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


def main() -> None:
    dry_run = os.environ.get("STOCK_DRY_RUN") == "1"
    days_ahead = int(os.environ.get("STOCK_DAYS_AHEAD", "90"))
    master_key = os.environ.get("SEMANTIK_MASTER_KEY", "")
    if not master_key:
        raise Stop("clé maître absente")
    work = pathlib.Path(os.environ.get("STOCK_WORK", "work")).resolve()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
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
    log("Données privées chargées")

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
    if json.loads((daily_dir / f"{today_number}.json").read_text()).get("vocab") != vocab_hash:
        raise Stop("le vocabulaire des données n'est pas le vocabulaire courant : republier les données (scripts/local-deploy.sh --publish-data)")
    manifest_path = work / "manifest.json"
    stock_first, stock_last = today_number, today_number - 1
    if storage.get(STOCK_PREFIX + "manifest.json", manifest_path):
        manifest = json.loads(manifest_path.read_text())
        stock_first, stock_last = manifest["first"], manifest["last"]
    existing = {}
    for number in range(max(stock_first, today_number), stock_last + 1):
        destination = daily_dir / f"{number}.json"
        if number == today_number and destination.exists():
            continue  # le jour courant en ligne fait foi
        if not storage.get(f"{STOCK_PREFIX}{number}.json", destination):
            raise Stop(f"jour n°{number} annoncé par le manifeste mais absent du stock")
    # Jours à venir déjà EN LIGNE (épinglés par le Deploy : leur mot ne doit pas changer) : repris tels quels s'ils sont lisibles (masqués au public : 404, alors régénérés)
    for number in (today_number + 1, today_number + 2):
        destination = daily_dir / f"{number}.json"
        if destination.exists():
            continue
        if site_get(f"daily/{number}.json", destination) == "erreur":
            raise Stop(f"jour n°{number} illisible en ligne")
    for path in daily_dir.glob("*.json"):
        existing[path.name] = sha(path)
    log(f"Historique chargé : {today_number - 1} jours passés, stock existant jusqu'au n°{stock_last}")
    candidates = work / "candidates.txt"
    candidates.write_text("\n".join(line.split("\t")[1] for line in (files / "candidates.tsv").read_text(encoding="utf-8").splitlines()[1:]) + "\n", encoding="utf-8")
    (work / "pinned").mkdir()
    environment = dict(os.environ, SEMANTIK_MASTER_KEY=master_key, PYTHONPATH=str(HERE / "pipeline"))
    if dry_run and (files / "start_from.txt").exists():
        # Mot de départ : AVANT toute suppression dans R2, on regénère à blanc les jours du stock concernés et on les compare à l'existant
        check_start_regeneration(work, data, daily_dir, files, models, candidates, environment, launch, today_number, stock_first, stock_last)
    if stock_last >= target_number:
        log("Stock déjà suffisant : rien à générer")
        return

    # 5. Génération (les jours existants sont conservés tels quels par le pipeline)
    command = pipeline_command(data, target_date, target_number, candidates, work, files, models)
    run_silent("génération des jours", command, cwd=work, env=environment)

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


if __name__ == "__main__":
    try:
        main()
    except Stop as stop:
        log(f"ARRÊT : {stop}")
        sys.exit(1)
    except Exception as error:  # noqa: BLE001 — le type seul, jamais le message ni la trace (ils pourraient contenir une donnée)
        log(f"ERREUR INATTENDUE : {type(error).__name__}")
        sys.exit(1)
