"""Tarea de Inspect: imaginar antes de continuar *Niebla* y medirlo contra el texto real.

Equivale a `imagine.evaluar`, pero con logs y visor de Inspect:

    uv run inspect eval imagine/inspect_tarea.py --model mlx/Qwen/Qwen3.5-0.8B-Base
    uv run inspect eval imagine/inspect_tarea.py --model mlx/runs/imagine300/paso_00300
"""

import json
import math
from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    metric,
    score_reducer,
    scorer,
)
from inspect_ai.solver import TaskState, generate

# Imports absolutos: Inspect carga este archivo como módulo suelto, no como parte del paquete.
from imagine import inspect_mlx
from imagine.modelo import CIERRA, cargar, prompt
from imagine.recompensas import formato_ok, log_verosimilitud

DATOS = Path(__file__).resolve().parents[1] / "data" / "test.jsonl"
_REFERENCIA = None


def _modelo_referencia():
    """Modelo base congelado que mide la recompensa (se carga una sola vez)."""
    global _REFERENCIA
    if _REFERENCIA is None:
        _REFERENCIA = inspect_mlx.en_hilo_mlx(cargar)
    return _REFERENCIA


def _media_sin_nan(valores) -> float:
    validos = [v for v in valores if not math.isnan(v)]
    return sum(validos) / len(validos) if validos else float("nan")


@metric
def media_valida() -> Metric:
    """Media que ignora las muestras sin valor (Δ no existe si el formato es inválido)."""

    def calcular(scores: list[SampleScore]) -> float:
        return _media_sin_nan([s.score.as_float() for s in scores])

    return calcular


@score_reducer(name="media_valida")
def media_valida_reducer():
    """Promedia las épocas de un fragmento clave por clave, ignorando los NaN."""

    def reducir(scores: list[Score]) -> Score:
        claves = scores[0].value.keys()
        return Score(value={k: _media_sin_nan([s.value[k] for s in scores]) for k in claves})

    return reducir


@scorer(
    metrics={
        "formato": [media_valida()],
        "recompensa": [media_valida()],
        "delta": [media_valida()],
        "ayuda": [media_valida()],
    }
)
def corpus() -> Scorer:
    """Formato de <imagine> y Δ log-verosimilitud del texto real del autor."""

    async def puntuar(state: TaskState, target: Target) -> Score:
        imaginado = state.output.completion
        cerro = (state.output.metadata or {}).get("parada") == CIERRA
        if not formato_ok(imaginado, cerro):
            nan = float("nan")
            return Score(
                value={"formato": 0.0, "recompensa": 0.0, "delta": nan, "ayuda": nan},
                answer=imaginado,
            )
        ref_model, tok = _modelo_referencia()

        def medir():
            p_ids = tok.encode(state.input_text)
            cuerpo = tok.encode(imaginado + CIERRA, add_special_tokens=False)
            (con,), sin = log_verosimilitud(ref_model, tok, p_ids, [cuerpo], target.text)
            return con - sin

        delta = await inspect_mlx.en_hilo_mlx_async(medir)
        return Score(
            value={
                "formato": 1.0,
                "recompensa": 1.0 + math.tanh(4 * delta),
                "delta": delta,
                "ayuda": float(delta > 0),
            },
            answer=imaginado,
            explanation=f"Δ = {delta:+.4f} nats/token",
        )

    return puntuar


@task
def niebla(n: int = 40, muestras: int = 4, temperatura: float = 0.8, max_tokens: int = 96):
    filas = [json.loads(l) for l in open(DATOS, encoding="utf-8")][:n]
    dataset = MemoryDataset(
        [
            Sample(
                id=i,
                input=prompt(f["contexto"]),
                target=f["referencia"],
                metadata={"libro": f["libro"]},
            )
            for i, f in enumerate(filas)
        ]
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        scorer=corpus(),
        epochs=Epochs(muestras, "media_valida"),
        config=GenerateConfig(
            temperature=temperatura, max_tokens=max_tokens, stop_seqs=[CIERRA]
        ),
    )
