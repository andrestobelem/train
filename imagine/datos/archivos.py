"""Dónde viven los datos del proyecto y cómo se leen y escriben.

Las rutas se resuelven contra la raíz del proyecto, no contra el directorio de
trabajo: Inspect, Gradio y los scripts se pueden lanzar desde cualquier lado.
"""

import json
from pathlib import Path
from typing import IO, Iterable

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "data"
TRAIN = DATOS / "train.jsonl"
TEST = DATOS / "test.jsonl"
SFT = DATOS / "sft.jsonl"


def ruta_del_proyecto(ruta: str | Path) -> Path:
    """Una ruta relativa se interpreta desde la raíz del proyecto."""
    ruta = Path(ruta)
    return ruta if ruta.is_absolute() else RAIZ / ruta


def leer_jsonl(ruta: str | Path, n: int | None = None) -> list[dict]:
    with open(ruta_del_proyecto(ruta), encoding="utf-8") as f:
        filas = [json.loads(linea) for linea in f]
    return filas[:n] if n is not None else filas


def escribir_jsonl(ruta: str | Path, filas: Iterable[dict]) -> None:
    with open(ruta_del_proyecto(ruta), "w", encoding="utf-8") as f:
        for fila in filas:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def agregar_jsonl(archivo: IO[str], fila: dict) -> None:
    """Agrega una fila a un archivo abierto y la deja en disco (logs y progreso)."""
    archivo.write(json.dumps(fila, ensure_ascii=False) + "\n")
    archivo.flush()
