"""Inspect Flow: evalúa en *Niebla* el modelo base y cada checkpoint de un entrenamiento.

    uv run flow run evals/niebla_flow.py
    uv run inspect view --log-dir logs/niebla

Los checkpoints se buscan en IMAGINE_RUN (por defecto runs/imagine300). Con el Flow
Store, volver a correrlo sólo evalúa los modelos que todavía no tienen log.
"""

import os
from pathlib import Path

from inspect_flow import FlowOptions, FlowSpec, FlowTask, tasks_matrix

RAIZ = Path(__file__).resolve().parents[1]
CORRIDA = Path(os.environ.get("IMAGINE_RUN", "runs/imagine300"))

modelos = ["mlx/Qwen/Qwen3.5-0.8B-Base"] + [
    f"mlx/{p}" for p in sorted((RAIZ / CORRIDA).glob("paso_*")) if p.is_dir()
]
# Rutas relativas a la raíz: son el nombre del modelo en los logs.
modelos = [m.replace(f"{RAIZ}/", "") for m in modelos]

FlowSpec(
    log_dir="../logs/niebla",
    # MLX usa una sola GPU: de a una tarea y de a una muestra.
    options=FlowOptions(max_tasks=1, max_samples=1, display="plain", log_dir_allow_dirty=True),
    tasks=tasks_matrix(
        task=FlowTask(name="imagine/niebla"),
        model=modelos,
    ),
)
