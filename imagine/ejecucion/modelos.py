"""Cargar modelos: el base de Hugging Face o el base más una carpeta de adaptadores."""

from mlx_lm import load

from imagine.datos.archivos import ruta_del_proyecto
from imagine.persistencia import adaptadores

MODELO_BASE = "Qwen/Qwen3.5-0.8B-Base"

_CACHE: dict[str, tuple] = {}


def cargar(adaptadores_dir: str | None = None, base: str = MODELO_BASE):
    """Una copia nueva del modelo (para entrenar: cada copia tiene sus pesos)."""
    if adaptadores_dir is not None:
        adaptadores_dir = str(ruta_del_proyecto(adaptadores_dir))
    return load(base, adapter_path=adaptadores_dir)


def resolver(nombre: str) -> tuple[str, str | None]:
    """Un nombre de modelo → (modelo base, carpeta de adaptadores o None).

    El nombre es un repo de Hugging Face (`Qwen/Qwen3.5-0.8B-Base`) o una carpeta de
    adaptadores del proyecto (`runs/sft/epoca_1`), cuyo config dice cuál es su base.
    """
    ruta = ruta_del_proyecto(nombre)
    if adaptadores.es_adaptador(ruta):
        return adaptadores.leer_config(ruta)["model"], str(ruta)
    return nombre, None


def cargar_por_nombre(nombre: str = MODELO_BASE):
    """El modelo para inferencia, cargado una sola vez por proceso y compartido."""
    if nombre not in _CACHE:
        base, adaptadores_dir = resolver(nombre)
        _CACHE[nombre] = cargar(adaptadores_dir, base=base)
    return _CACHE[nombre]
