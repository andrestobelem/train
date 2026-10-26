"""Métrica y reductor de Inspect compartidos por las tareas del proyecto.

Algunos valores no existen para todas las muestras (por ejemplo Δ cuando el formato
es inválido); se marcan con NaN y se ignoran al promediar.
"""

import math

from inspect_ai.scorer import Metric, SampleScore, Score, metric, score_reducer

NAN = float("nan")


def media_sin_nan(valores) -> float:
    validos = [v for v in valores if not math.isnan(v)]
    return sum(validos) / len(validos) if validos else NAN


@metric
def media_valida() -> Metric:
    """Media que ignora las muestras sin valor."""

    def calcular(scores: list[SampleScore]) -> float:
        return media_sin_nan([s.score.as_float() for s in scores])

    return calcular


@score_reducer(name="media_valida")
def media_valida_reducer():
    """Promedia las épocas de un fragmento clave por clave, ignorando los NaN."""

    def reducir(scores: list[Score]) -> Score:
        claves = scores[0].value.keys()
        return Score(value={k: media_sin_nan([s.value[k] for s in scores]) for k in claves})

    return reducir
