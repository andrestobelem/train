"""Adaptadores LoRA en disco, en el formato que entiende `mlx_lm.load(adapter_path=…)`.

Una carpeta de adaptadores tiene `adapters.safetensors` y `adapter_config.json`. Al
config de mlx-lm se le agrega `model`: el modelo base sobre el que se entrenaron.
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

CONFIG = "adapter_config.json"
PESOS = "adapters.safetensors"

# Así empieza un archivo que Git LFS todavía no bajó: en su lugar hay un puntero de texto.
PUNTERO_LFS = b"version https://git-lfs"


def config_lora(base: str, capas: int, rango: int, escala: float) -> dict:
    return {
        "fine_tune_type": "lora",
        "num_layers": capas,
        "lora_parameters": {"rank": rango, "scale": escala, "dropout": 0.0},
        "model": base,
    }


def guardar(model, destino: Path, config: dict) -> None:
    """Guarda los parámetros entrenables del modelo (las matrices LoRA) y su config."""
    destino.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(destino / PESOS), dict(tree_flatten(model.trainable_parameters())))
    (destino / CONFIG).write_text(json.dumps(config, indent=2))


def es_adaptador(ruta: Path) -> bool:
    return (ruta / CONFIG).exists()


def sin_bajar(ruta: Path) -> bool:
    """Si los pesos son todavía un puntero de Git LFS: el repo se clonó sin bajarlos."""
    pesos = ruta / PESOS
    if not pesos.exists():
        return False
    with open(pesos, "rb") as f:
        return f.read(len(PUNTERO_LFS)) == PUNTERO_LFS


def leer_config(ruta: Path) -> dict:
    return json.loads((ruta / CONFIG).read_text())
