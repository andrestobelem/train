"""Elegir el modelo desde la interfaz: avisar con claridad si no está y cómo conseguirlo."""

import sys

from imagine.datos.archivos import ruta_del_proyecto
from imagine.ejecucion.modelos import MODELO_BASE, cargar_por_nombre
from imagine.persistencia import adaptadores, catalogo

COMO_BAJAR_DE_LFS = """Los modelos vienen con el repo por Git LFS, pero no se bajaron.

  brew install git-lfs
  git lfs install
  git lfs pull"""

COMO_BAJAR_EL_FINAL = f"""No encuentro el modelo final en {catalogo.FINAL}.

Viene con el repo por Git LFS. Si clonaste sin LFS:

  git lfs install
  git lfs pull

También está en Hugging Face:

  uv run hf download {catalogo.REPO_FINAL} --local-dir {catalogo.FINAL}"""


def disponible(nombre: str) -> bool:
    """Los modelos del proyecto están en runs/; el resto se baja de Hugging Face al cargarlo."""
    if not nombre.startswith("runs/"):
        return True
    ruta = ruta_del_proyecto(nombre)
    return adaptadores.es_adaptador(ruta) and not adaptadores.sin_bajar(ruta)


def falta(nombre: str) -> str | None:
    """Si el modelo no está en esta máquina, qué hacer para conseguirlo."""
    if disponible(nombre):
        return None
    if adaptadores.sin_bajar(ruta_del_proyecto(nombre)):
        return COMO_BAJAR_DE_LFS
    if nombre == catalogo.FINAL:
        return COMO_BAJAR_EL_FINAL
    return f"No encuentro el modelo {nombre}. Se entrena con docs/entrenar-y-evaluar.md."


def cargar(nombre: str):
    """El modelo para la línea de comandos; si no está, explica cómo conseguirlo y sale."""
    if aviso := falta(nombre):
        sys.exit(f"{aviso}\n\nO probá con el modelo sin entrenar: --modelo {MODELO_BASE}")
    print("Cargando el modelo…", file=sys.stderr)
    return cargar_por_nombre(nombre)
