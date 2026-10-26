"""Inspect Flow: calidad de la prosa del modelo base y de los modelos entrenados.

    uv run flow run evals/prosa_flow.py
    uv run inspect view --log-dir logs/prosa

Compara el modelo base escribiendo directamente, el modelo base imaginando antes y
los adaptadores listados en IMAGINE_MODELOS (separados por comas).
"""

import os

from inspect_flow import FlowOptions, FlowSpec, FlowTask

TAREA = "../imagine/inspect_prosa.py@prosa"
BASE = "mlx/Qwen/Qwen3.5-0.8B-Base"
ENTRENADOS = os.environ.get(
    "IMAGINE_MODELOS", "runs/imagine300/paso_00150,runs/imagine300/paso_00300"
).split(",")

FlowSpec(
    log_dir="../logs/prosa",
    # MLX usa una sola GPU: de a una tarea y de a una muestra.
    options=FlowOptions(
        max_tasks=1, max_samples=1, display="plain", log_dir_allow_dirty=True
    ),
    tasks=[
        FlowTask(name=TAREA, model=BASE, args={"imaginar": False}),
        FlowTask(name=TAREA, model=BASE, args={"imaginar": True}),
    ]
    + [FlowTask(name=TAREA, model=f"mlx/{m.strip()}") for m in ENTRENADOS if m.strip()],
)
