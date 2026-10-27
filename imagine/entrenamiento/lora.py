"""Qué parámetros se entrenan: sólo las matrices LoRA, el modelo base queda congelado."""

from mlx.utils import tree_flatten
from mlx_lm.tuner.utils import linear_to_lora_layers


def agregar_lora(model, capas: int, rango: int, escala: float) -> None:
    """Congela el modelo y le agrega adaptadores LoRA nuevos (inicializados en cero)."""
    model.freeze()
    linear_to_lora_layers(model, capas, {"rank": rango, "scale": escala, "dropout": 0.0})


def entrenar_solo_lora(model) -> None:
    """Para seguir entrenando adaptadores ya cargados: sólo lora_a y lora_b."""
    model.freeze()
    model.apply_to_modules(
        lambda _, m: (
            m.unfreeze(keys=["lora_a", "lora_b"], strict=False) if hasattr(m, "lora_a") else None
        )
    )


def millones_entrenables(model) -> float:
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters())) / 1e6
