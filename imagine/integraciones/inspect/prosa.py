"""Tarea `imagine/prosa`: calidad de la continuación que se escribe después de imaginar.

`imagine/niebla` mide si lo imaginado anticipa al autor; esta tarea mide lo que el
lector efectivamente lee. Métricas (todas por reglas, sin juez):

- repeticion, distintos, ajenos, palabras: ver `imagine.dominio.prosa`.
- fluidez: log-prob media por token de la continuación según el modelo base, dado el
  fragmento sin la imaginación (penaliza lo que sigue a la imaginación y no al fragmento).
- chrf: similitud de n-gramas de caracteres con lo que escribió el autor (0 a 1).
"""

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from sacrebleu.metrics import CHRF

from imagine.dominio import prosa as reglas
from imagine.dominio.escritura import (
    CIERRA,
    FINES_CONTINUACION,
    TEMPERATURA_MUESTREO,
    TOKENS_IMAGINACION,
    cortar_continuacion,
    formato_ok,
    prompt_continuacion,
)
from imagine.ejecucion import puntuacion
from imagine.ejecucion.hilo import en_hilo_mlx, en_hilo_mlx_async
from imagine.ejecucion.modelos import cargar_por_nombre

from .metricas import NAN, media_valida
from .niebla import dataset

METRICAS = ["formato", "repeticion", "distintos", "ajenos", "fluidez", "chrf", "palabras"]
_CHRF = CHRF()


@solver
def imaginar_y_continuar(imaginar: bool, max_tokens: int, temperatura: float) -> Solver:
    async def resolver(state: TaskState, generate: Generate) -> TaskState:
        fragmento = state.metadata["fragmento"]
        if imaginar:
            # Usa la config de la tarea: corta en </imagine>.
            state = await generate(state)
            imaginado = state.output.completion
            cerro = (state.output.metadata or {}).get("parada") == CIERRA
        else:
            imaginado, cerro = "", True
        state.store.set("imaginado", imaginado)
        state.store.set("formato", formato_ok(imaginado, cerro) if imaginar else True)

        salida = await get_model().generate(
            prompt_continuacion(fragmento, imaginado),
            config=GenerateConfig(
                max_tokens=max_tokens,
                temperature=temperatura,
                stop_seqs=list(FINES_CONTINUACION),
            ),
        )
        state.output = salida
        state.store.set("continuacion", cortar_continuacion(salida.completion))
        return state

    return resolver


@scorer(metrics={k: [media_valida()] for k in METRICAS})
def calidad() -> Scorer:
    async def puntuar(state: TaskState, target: Target) -> Score:
        texto = state.store.get("continuacion", "")
        ref_model, tok = en_hilo_mlx(cargar_por_nombre)

        def fluidez():
            if not texto:
                return NAN
            return puntuacion.fluidez(ref_model, tok, tok.encode(state.input_text), texto)

        return Score(
            value={
                "formato": float(state.store.get("formato")),
                "repeticion": reglas.repeticion(texto),
                "distintos": reglas.distintos(texto),
                "ajenos": reglas.ajenos(texto),
                "fluidez": await en_hilo_mlx_async(fluidez),
                "chrf": _CHRF.sentence_score(texto, [target.text]).score / 100,
                "palabras": float(reglas.palabras(texto)),
            },
            answer=texto,
            metadata={"imaginado": state.store.get("imaginado")},
        )

    return puntuar


@task
def prosa(
    n: int = 40,
    muestras: int = 2,
    imaginar: bool = True,
    temperatura: float = TEMPERATURA_MUESTREO,
    max_tokens_imaginar: int = TOKENS_IMAGINACION,
    max_tokens_continuar: int = 128,
):
    return Task(
        dataset=dataset(n),
        solver=imaginar_y_continuar(imaginar, max_tokens_continuar, temperatura),
        scorer=calidad(),
        epochs=Epochs(muestras, "media_valida"),
        config=GenerateConfig(
            temperature=temperatura, max_tokens=max_tokens_imaginar, stop_seqs=[CIERRA]
        ),
    )
