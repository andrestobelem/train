"""Tarea `imagine/prosa`: calidad de la continuación que se escribe después de imaginar.

`imagine/niebla` mide si la imaginación anticipa la referencia; esta tarea mide lo que el
lector efectivamente lee. Métricas (todas por reglas, sin juez):

- repeticion, distintos, letras_ajenas, palabras: ver `imagine.dominio.prosa`.
- fluidez: log-prob media por token de la continuación según el modelo base, dado el
  fragmento sin la imaginación (penaliza lo que sigue a la imaginación y no al fragmento).
- parecido (con el autor): chrF contra la referencia, de 0 a 1 (ver GLOSSARY.md).
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
    SIN_IMAGINACION,
    TEMPERATURA_MUESTREO,
    TOKENS_IMAGINACION,
    cortar_continuacion,
    prompt_continuacion,
)
from imagine.ejecucion import puntuacion
from imagine.ejecucion.hilo import en_hilo_mlx, en_hilo_mlx_async
from imagine.ejecucion.modelos import cargar_por_nombre

from . import dataset
from .metricas import NAN, media_valida
from .proveedor import imaginacion_de

METRICAS = [
    "formato",
    "repeticion",
    "distintos",
    "letras_ajenas",
    "fluidez",
    "parecido",
    "palabras",
]
_CHRF = CHRF()


@solver
def imaginar_y_continuar(imaginar: bool, max_tokens: int, temperatura: float) -> Solver:
    async def resolver(state: TaskState, generate: Generate) -> TaskState:
        fragmento = state.metadata["fragmento"]
        imaginacion = SIN_IMAGINACION
        if imaginar:
            # Usa la config de la tarea: corta en </imagine>.
            state = await generate(state)
            imaginacion = imaginacion_de(state.output)
        state.store.set("imaginacion", imaginacion.texto)
        state.store.set("formato", imaginacion.formato_valido if imaginar else True)

        salida = await get_model().generate(
            prompt_continuacion(fragmento, imaginacion),
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
        juez, tok = en_hilo_mlx(cargar_por_nombre)

        def fluidez():
            if not texto:
                return NAN
            return puntuacion.fluidez(juez, tok, tok.encode(state.input_text), texto)

        return Score(
            value={
                "formato": float(state.store.get("formato")),
                "repeticion": reglas.repeticion(texto),
                "distintos": reglas.distintos(texto),
                "letras_ajenas": reglas.letras_ajenas(texto),
                "fluidez": await en_hilo_mlx_async(fluidez),
                "parecido": _CHRF.sentence_score(texto, [target.text]).score / 100,
                "palabras": float(reglas.palabras(texto)),
            },
            answer=texto,
            metadata={"imaginacion": state.store.get("imaginacion")},
        )

    return puntuar


@task
def prosa(
    n: int = 40,
    intentos: int = 2,
    imaginar: bool = True,
    temperatura: float = TEMPERATURA_MUESTREO,
    max_tokens_imaginar: int = TOKENS_IMAGINACION,
    max_tokens_continuar: int = 128,
):
    return Task(
        dataset=dataset.niebla(n),
        solver=imaginar_y_continuar(imaginar, max_tokens_continuar, temperatura),
        scorer=calidad(),
        epochs=Epochs(intentos, "media_valida"),
        config=GenerateConfig(
            temperature=temperatura, max_tokens=max_tokens_imaginar, stop_seqs=[CIERRA]
        ),
    )
