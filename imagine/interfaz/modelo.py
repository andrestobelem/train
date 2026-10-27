"""Elegir el modelo desde la interfaz: avisar con claridad si no está y cómo conseguirlo."""

import sys

from imagine.datos.archivos import ruta_del_proyecto
from imagine.ejecucion.modelos import MODELO_BASE, cargar_por_nombre
from imagine.persistencia import adaptadores, catalogo

COMO_BAJAR_EL_FINAL = f"""No encuentro el modelo final en {catalogo.FINAL}.

Bajalo de Hugging Face (hace falta una cuenta con acceso al repo):

  uv run hf auth login
  uv run hf download {catalogo.REPO_FINAL} --local-dir {catalogo.FINAL}"""


def disponible(nombre: str) -> bool:
    """Los modelos del proyecto están en runs/; el resto se baja de Hugging Face al cargarlo."""
    if not nombre.startswith("runs/"):
        return True
    return adaptadores.es_adaptador(ruta_del_proyecto(nombre))


def falta(nombre: str) -> str | None:
    """Si el modelo no está en esta máquina, qué hacer para conseguirlo."""
    if disponible(nombre):
        return None
    if nombre == catalogo.FINAL:
        return COMO_BAJAR_EL_FINAL
    return f"No encuentro el modelo {nombre}. Se entrena con docs/entrenar-y-evaluar.md."


def cargar(nombre: str):
    """El modelo para la línea de comandos; si no está, explica cómo conseguirlo y sale."""
    if aviso := falta(nombre):
        sys.exit(f"{aviso}\n\nO probá con el modelo sin entrenar: --modelo {MODELO_BASE}")
    print("Cargando el modelo…", file=sys.stderr)
    return cargar_por_nombre(nombre)
