"""Integración con Inspect AI: proveedor `mlx/` y tareas `imagine/niebla` e `imagine/prosa`.

Inspect importa este paquete por el entry point de pyproject.toml, lo que registra
todo con el prefijo `imagine/`:

    uv run inspect eval imagine/niebla --model mlx/runs/sft/epoca_1
    uv run inspect eval imagine/prosa --model mlx/Qwen/Qwen3.5-0.8B-Base -T imaginar=false
"""

from . import metricas, proveedor
from .niebla import niebla
from .prosa import prosa

__all__ = ["metricas", "proveedor", "niebla", "prosa"]
