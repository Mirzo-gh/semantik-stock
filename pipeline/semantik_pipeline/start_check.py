"""Comparaison « mêmes mots, mêmes scores, plus start » entre deux dossiers de jours (avant / après une régénération du stock).

N'affiche ni ne renvoie jamais un mot ni un index : seulement des compteurs et des messages d'écart sans donnée. Autonome (masking + base64) :
utilisé par pipeline/check_start_added.py et par stock_run.py (dry_run du dépôt public).
"""
import base64
import json
import pathlib

import numpy as np

from .masking import xor_mask

BAND_MIN, BAND_MAX = 5600, 6400  # 26-34 °C : la bande la plus large du choix


def _decode(puzzle: dict, field: str) -> np.ndarray:
    return np.frombuffer(xor_mask(base64.b64decode(puzzle[field]), puzzle["seed"]), dtype="<u2")


def compare_days(before_dir, after_dir, first: int = 1) -> tuple[dict, list[str]]:
    """({'days', 'compared', 'with_start'}, problèmes). `first` = premier jour qui doit avoir un `start`.

    Pour chaque jour de `after_dir` : s'il existe dans `before_dir`, tous les champs hors `start` sont identiques (et `start` aussi s'il était déjà là) ;
    à partir de `first`, `start` est présent, dans la bande 26-34 °C, hors du top 1000 et différent du mot du jour.
    """
    before_dir, after_dir = pathlib.Path(before_dir), pathlib.Path(after_dir)
    numbers = sorted(int(path.stem) for path in after_dir.glob("*.json") if path.stem.isdigit())
    problems: list[str] = []
    compared = with_start = 0
    for number in numbers:
        after = json.loads((after_dir / f"{number}.json").read_text())
        before_path = before_dir / f"{number}.json"
        if before_path.exists():
            before = json.loads(before_path.read_text())
            if {key: value for key, value in after.items() if key != "start"} != {key: value for key, value in before.items() if key != "start"}:
                problems.append(f"n°{number} : un champ autre que `start` a changé")
            if "start" in before and before.get("start") != after.get("start"):
                problems.append(f"n°{number} : `start` différent de celui déjà en stock")
            compared += 1
        if "start" not in after:
            if number >= first:
                problems.append(f"n°{number} : `start` absent")
            continue
        with_start += 1
        if number < first:
            problems.append(f"n°{number} : `start` présent avant le premier jour concerné")
        scores, top = _decode(after, "scores"), _decode(after, "top")
        start = int.from_bytes(xor_mask(base64.b64decode(after["start"]), after["seed"]), "little")
        if not 0 <= start < len(scores):
            problems.append(f"n°{number} : index de départ hors vocabulaire")
        elif not BAND_MIN <= int(scores[start]) <= BAND_MAX or start in set(top.tolist()) or start == int(top[-1]):
            problems.append(f"n°{number} : mot de départ hors bande 26-34 °C, dans le top ou égal au mot du jour")
    return {"days": len(numbers), "compared": compared, "with_start": with_start}, problems
