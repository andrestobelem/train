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
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState, generate

# Imports absolutos: Inspect carga este archivo como módulo suelto, no como parte del paquete.
from imagine import inspect_mlx
from imagine.inspect_metricas import media_valida
from imagine.modelo import CIERRA, prompt
from imagine.recompensas import formato_ok, log_verosimilitud

DATOS = Path(__file__).resolve().parents[1] / "data" / "test.jsonl"


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
        ref_model, tok = inspect_mlx.modelo_referencia()

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
