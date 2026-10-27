"""Inspect Flow: calidad de la prosa del modelo base y de los modelos entrenados.

    uv run flow run evals/prosa_flow.py
    uv run inspect view --log-dir logs/prosa

Compara el modelo base escribiendo directamente, el modelo base imaginando antes y
los modelos listados en IMAGINE_MODELOS (separados por comas; por defecto, el de RL y
el final del catálogo).
"""

import os

from inspect_flow import FlowSpec, FlowTask

from imagine.ejecucion.modelos import MODELO_BASE
from imagine.persistencia import catalogo

TAREA = "imagine/prosa"
entrenados = os.environ.get("IMAGINE_MODELOS", f"{catalogo.RL},{catalogo.FINAL}").split(",")

FlowSpec(
    log_dir="../logs/prosa",
    tasks=[
        FlowTask(name=TAREA, model=f"mlx/{MODELO_BASE}", args={"imaginar": False}),
        FlowTask(name=TAREA, model=f"mlx/{MODELO_BASE}", args={"imaginar": True}),
    ]
    + [FlowTask(name=TAREA, model=f"mlx/{m.strip()}") for m in entrenados if m.strip()],
)
