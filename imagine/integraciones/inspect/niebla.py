"""Tarea `imagine/niebla`: imaginar antes de continuar *Niebla* y medirlo contra el autor."""

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState, generate

from imagine.dominio.escritura import (
    CIERRA,
    TEMPERATURA_MUESTREO,
    TOKENS_IMAGINACION,
    formato_ok,
)
from imagine.dominio.recompensa import recompensa
from imagine.ejecucion import puntuacion
from imagine.ejecucion.hilo import en_hilo_mlx, en_hilo_mlx_async
from imagine.ejecucion.modelos import cargar_por_nombre

from . import dataset
from .metricas import NAN, media_valida


@scorer(metrics={k: [media_valida()] for k in ["formato", "recompensa", "delta", "ayuda"]})
def corpus() -> Scorer:
    """Formato de <imagine> y Δ log-verosimilitud del texto real del autor."""

    async def puntuar(state: TaskState, target: Target) -> Score:
        imaginado = state.output.completion
        cerro = (state.output.metadata or {}).get("parada") == CIERRA
        if not formato_ok(imaginado, cerro):
            valor = {"formato": 0.0, "recompensa": recompensa(None), "delta": NAN, "ayuda": NAN}
            return Score(value=valor, answer=imaginado)
        ref_model, tok = en_hilo_mlx(cargar_por_nombre)

        def medir():
            p_ids = tok.encode(state.input_text)
            (delta,) = puntuacion.deltas(ref_model, tok, p_ids, [(imaginado, True)], target.text)
            return delta

        delta = await en_hilo_mlx_async(medir)
        return Score(
            value={
                "formato": 1.0,
                "recompensa": recompensa(delta),
                "delta": delta,
                "ayuda": float(delta > 0),
            },
            answer=imaginado,
            explanation=f"Δ = {delta:+.4f} nats/token",
        )

    return puntuar


@task
def niebla(
    n: int = 40,
    muestras: int = 4,
    temperatura: float = TEMPERATURA_MUESTREO,
    max_tokens: int = TOKENS_IMAGINACION,
):
    return Task(
        dataset=dataset.niebla(n),
        solver=generate(),
        scorer=corpus(),
        epochs=Epochs(muestras, "media_valida"),
        config=GenerateConfig(temperature=temperatura, max_tokens=max_tokens, stop_seqs=[CIERRA]),
    )
