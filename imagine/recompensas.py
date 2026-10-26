"""Recompensas basadas en reglas, al estilo DeepSeek-R1.

R1 usa dos recompensas: exactitud (¿la respuesta final es correcta?) y formato
(¿el razonamiento está entre <think> </think>?). Acá no hay una respuesta "correcta",
pero sí un corpus literario: sabemos qué escribió el autor a continuación.

- Formato: lo imaginado tiene que cerrar con </imagine>, no estar vacío y no
  reabrir la etiqueta.
- Corpus: cuánto mejora lo imaginado la predicción del texto real del autor.
  Se mide con el modelo de referencia congelado como log p(referencia | contexto,
  imaginación) − log p(referencia | contexto, imaginación vacía), promediado por
  token. Es una recompensa verificable contra el corpus: la imaginación no ve la
  referencia, sólo gana si anticipa lo que de verdad pasa en la novela.
"""

import math

import mlx.core as mx

from .modelo import ABRE, CIERRA, logprobs, rellenar

MIN_PALABRAS = 5


def formato_ok(imaginado: str, cerro: bool) -> bool:
    return cerro and ABRE not in imaginado and len(imaginado.split()) >= MIN_PALABRAS


def log_verosimilitud(ref_model, tok, prompt_ids, imaginaciones_ids, referencia: str):
    """Log-prob media por token de la referencia tras cada imaginación (y sin ninguna).

    Devuelve (lista por imaginación, valor sin imaginación).
    """
    cierre = tok.encode(CIERRA, add_special_tokens=False)
    ref_ids = tok.encode("\n" + referencia, add_special_tokens=False)
    # La última fila es la línea de base: imaginación vacía.
    cuerpos = [list(c) for c in imaginaciones_ids] + [cierre]
    seqs = [prompt_ids + c + ref_ids for c in cuerpos]
    inicios = mx.array([len(prompt_ids) + len(c) for c in cuerpos])
    pos = inicios[:, None] + mx.arange(len(ref_ids))[None]
    lp = logprobs(ref_model, rellenar(seqs, tok.eos_token_id), pos).mean(axis=1)
    lp = lp.tolist()
    return lp[:-1], lp[-1]


def recompensas(ref_model, tok, prompt_ids, completaciones, imaginados, referencia):
    """Recompensa total por completación y métricas para el log.

    Inválida → 0. Válida → 1 + tanh(4·Δ), con Δ en nats/token. Así una
    imaginación con formato correcto siempre supera a una sin formato, y entre
    las válidas manda cuánto ayudan a predecir al autor.
    """
    validas = [i for i, (txt, cerro) in enumerate(imaginados) if formato_ok(txt, cerro)]
    deltas = [None] * len(completaciones)
    if validas:
        # Se puntúa sólo lo que está dentro de la etiqueta (más el cierre).
        cierre = tok.encode(CIERRA, add_special_tokens=False)
        cuerpos = [
            tok.encode(imaginados[i][0], add_special_tokens=False) + cierre for i in validas
        ]
        con, sin = log_verosimilitud(ref_model, tok, prompt_ids, cuerpos, referencia)
        for i, lp in zip(validas, con):
            deltas[i] = lp - sin
    r = [0.0 if d is None else 1.0 + math.tanh(4 * d) for d in deltas]
    return r, deltas
