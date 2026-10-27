"""Tarea `imagine/niebla`: imaginar antes de continuar *Niebla* y medirlo contra el autor."""

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import TaskState, generate

from imagine.dominio.escritura import CIERRA, TEMPERATURA_MUESTREO, TOKENS_IMAGINACION
from imagine.dominio.recompensa import ayuda, recompensa
from imagine.ejecucion import puntuacion
from imagine.ejecucion.hilo import en_hilo_mlx, en_hilo_mlx_async
from imagine.ejecucion.modelos import cargar_por_nombre

from . import dataset
from .metricas import NAN, media_valida
from .proveedor import imaginacion_de


@scorer(metrics={k: [media_valida()] for k in ["formato", "recompensa", "ganancia", "ayuda"]})
def corpus() -> Scorer:
    """Formato de la imaginación y su ganancia sobre la referencia."""

    async def puntuar(state: TaskState, target: Target) -> Score:
        imaginacion = imaginacion_de(state.output)
        if not imaginacion.formato_valido:
            valor = {"formato": 0.0, "recompensa": recompensa(None), "ganancia": NAN, "ayuda": NAN}
            return Score(value=valor, answer=imaginacion.texto)
        juez, tok = en_hilo_mlx(cargar_por_nombre)

        def medir():
            p_ids = tok.encode(state.input_text)
            (ganancia,) = puntuacion.ganancias(juez, tok, p_ids, [imaginacion], target.text)
            return ganancia

        ganancia = await en_hilo_mlx_async(medir)
        return Score(
            value={
                "formato": 1.0,
                "recompensa": recompensa(ganancia),
                "ganancia": ganancia,
                "ayuda": float(ayuda(ganancia)),
            },
            answer=imaginacion.texto,
            explanation=f"Δ = {ganancia:+.4f} nats/token",
        )

    return puntuar


@task
def niebla(
    n: int = 40,
    intentos: int = 4,
    temperatura: float = TEMPERATURA_MUESTREO,
    max_tokens: int = TOKENS_IMAGINACION,
):
    return Task(
        dataset=dataset.niebla(n),
        solver=generate(),
        scorer=corpus(),
        epochs=Epochs(intentos, "media_valida"),
        config=GenerateConfig(temperature=temperatura, max_tokens=max_tokens, stop_seqs=[CIERRA]),
    )
