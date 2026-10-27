"""Inspect Flow: evalúa en *Niebla* el modelo base y cada checkpoint de una corrida de GRPO.

    uv run flow run evals/niebla_flow.py
    uv run inspect view --log-dir logs/niebla

Los checkpoints se buscan en IMAGINE_RUN (por defecto, la corrida de GRPO del catálogo).
"""

import os

from inspect_flow import FlowSpec, FlowTask, tasks_matrix

from imagine.datos.archivos import RAIZ, ruta_del_proyecto
from imagine.ejecucion.modelos import MODELO_BASE
from imagine.persistencia import catalogo

corrida = ruta_del_proyecto(os.environ.get("IMAGINE_RUN", catalogo.CORRIDA_GRPO))
# Rutas relativas a la raíz: son el nombre del modelo en los logs.
checkpoints = [p.relative_to(RAIZ) for p in sorted(corrida.glob("paso_*")) if p.is_dir()]

FlowSpec(
    log_dir="../logs/niebla",
    tasks=tasks_matrix(
        task=FlowTask(name="imagine/niebla"),
        model=[f"mlx/{MODELO_BASE}"] + [f"mlx/{p}" for p in checkpoints],
    ),
)
