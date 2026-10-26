"""Tarea de Inspect: calidad de la prosa que el modelo escribe después de imaginar.

`niebla` mide si lo imaginado anticipa al autor. Esta tarea mide lo que el lector
efectivamente lee: la continuación escrita después de </imagine>.

    uv run inspect eval imagine/inspect_prosa.py@prosa --model mlx/runs/imagine300/paso_00150
    uv run inspect eval imagine/inspect_prosa.py@prosa -T imaginar=false --model mlx/Qwen/Qwen3.5-0.8B-Base

Métricas (todas por reglas, sin juez):
- repeticion: fracción de trigramas de palabras repetidos (menos es mejor).
- distintos: fracción de bigramas de palabras distintos (más es mejor).
- ajenos: fracción de letras fuera del alfabeto latino, p. ej. griego o CJK (menos es mejor).
- fluidez: log-prob media por token de la continuación según el modelo base
  congelado, dado el fragmento (más es mejor; ojo, el texto repetitivo también es "fluido").
- chrf: similitud de n-gramas de caracteres con lo que escribió el autor (0 a 1).
"""

import json
import unicodedata
from pathlib import Path

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from sacrebleu.metrics import CHRF

# Imports absolutos: Inspect carga este archivo como módulo suelto, no como parte del paquete.
from imagine import inspect_mlx
from imagine.inspect_metricas import media_valida
from imagine.modelo import ABRE, CIERRA, prompt
from imagine.recompensas import formato_ok, log_verosimilitud

DATOS = Path(__file__).resolve().parents[1] / "data" / "test.jsonl"
# Donde el modelo empezaría otro turno de la conversación o volvería a las etiquetas.
FIN_CONTINUACION = ["\nLector:", ABRE, CIERRA]
_CHRF = CHRF()


@solver
def imaginar_y_continuar(imaginar: bool, max_tokens: int, temperatura: float) -> Solver:
    async def resolver(state: TaskState, generate: Generate) -> TaskState:
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
            state.input_text + imaginado + CIERRA + "\n",
            config=GenerateConfig(
                max_tokens=max_tokens, temperature=temperatura, stop_seqs=FIN_CONTINUACION
            ),
        )
        state.output = salida
        state.store.set("continuacion", salida.completion.strip())
        return state

    return resolver


def repeticion(palabras: list[str], n: int = 3) -> float:
    ngramas = [tuple(palabras[i : i + n]) for i in range(len(palabras) - n + 1)]
    return 1 - len(set(ngramas)) / len(ngramas) if ngramas else 0.0


def distintos(palabras: list[str], n: int = 2) -> float:
    ngramas = [tuple(palabras[i : i + n]) for i in range(len(palabras) - n + 1)]
    return len(set(ngramas)) / len(ngramas) if ngramas else 0.0


def ajenos(texto: str) -> float:
    letras = [c for c in texto if c.isalpha()]
    if not letras:
        return 0.0
    return sum(not unicodedata.name(c, "").startswith("LATIN") for c in letras) / len(letras)


@scorer(
    metrics={
        k: [media_valida()]
        for k in ["formato", "repeticion", "distintos", "ajenos", "fluidez", "chrf", "palabras"]
    }
)
def prosa_scorer() -> Scorer:
    async def puntuar(state: TaskState, target: Target) -> Score:
        texto = state.store.get("continuacion", "")
        palabras = texto.lower().split()
        ref_model, tok = inspect_mlx.modelo_referencia()

        def fluidez():
            if not texto:
                return float("nan")
            # Sin la imaginación: se mide la continuación como texto que sigue al fragmento.
            p_ids = tok.encode(state.input_text)
            _, lp = log_verosimilitud(ref_model, tok, p_ids, [], texto)
            return lp

        return Score(
            value={
                "formato": float(state.store.get("formato")),
                "repeticion": repeticion(palabras),
                "distintos": distintos(palabras),
                "ajenos": ajenos(texto),
                "fluidez": await inspect_mlx.en_hilo_mlx_async(fluidez),
                "chrf": _CHRF.sentence_score(texto, [target.text]).score / 100,
                "palabras": float(len(palabras)),
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
    temperatura: float = 0.8,
    max_tokens_imaginar: int = 96,
    max_tokens_continuar: int = 128,
):
    filas = [json.loads(l) for l in open(DATOS, encoding="utf-8")][:n]
    dataset = MemoryDataset(
        [
            Sample(id=i, input=prompt(f["contexto"]), target=f["referencia"])
            for i, f in enumerate(filas)
        ]
    )
    return Task(
        dataset=dataset,
        solver=imaginar_y_continuar(imaginar, max_tokens_continuar, temperatura),
        scorer=prosa_scorer(),
        epochs=Epochs(muestras, "media_valida"),
        config=GenerateConfig(
            temperature=temperatura, max_tokens=max_tokens_imaginar, stop_seqs=[CIERRA]
        ),
    )
