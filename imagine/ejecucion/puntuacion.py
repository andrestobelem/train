"""Puntuar contra el corpus con el modelo de referencia congelado.

Implementa las medidas que necesitan un modelo: Δ (cuánto ayuda lo imaginado a
predecir al autor) y la fluidez de una continuación. Las reglas que las usan
(formato, recompensa) están en `imagine.dominio`.
"""

import mlx.core as mx

from imagine.dominio.escritura import CIERRA, formato_ok

from .lm import logprobs, rellenar


def verosimilitud(ref_model, tok, prompt_ids, cuerpos_ids, texto: str):
    """Log-prob media por token de `texto` tras cada cuerpo, y tras una imaginación vacía.

    Cada cuerpo es lo imaginado más </imagine>. Devuelve (lista por cuerpo, valor vacío).
    """
    cierre = tok.encode(CIERRA, add_special_tokens=False)
    texto_ids = tok.encode("\n" + texto, add_special_tokens=False)
    # La última fila es la línea de base: imaginación vacía.
    cuerpos = [list(c) for c in cuerpos_ids] + [cierre]
    seqs = [prompt_ids + c + texto_ids for c in cuerpos]
    inicios = mx.array([len(prompt_ids) + len(c) for c in cuerpos])
    pos = inicios[:, None] + mx.arange(len(texto_ids))[None]
    lp = logprobs(ref_model, rellenar(seqs, tok.eos_token_id), pos).mean(axis=1).tolist()
    return lp[:-1], lp[-1]


def deltas(ref_model, tok, prompt_ids, imaginaciones, referencia: str) -> list[float | None]:
    """Δ en nats/token de cada imaginación (texto, cerró); None si el formato es inválido.

    Se puntúa sólo lo que está dentro de la etiqueta (más el cierre), re-tokenizado.
    """
    validas = [i for i, (texto, cerro) in enumerate(imaginaciones) if formato_ok(texto, cerro)]
    salida: list[float | None] = [None] * len(imaginaciones)
    if not validas:
        return salida
    cierre = tok.encode(CIERRA, add_special_tokens=False)
    cuerpos = [tok.encode(imaginaciones[i][0], add_special_tokens=False) + cierre for i in validas]
    con, sin = verosimilitud(ref_model, tok, prompt_ids, cuerpos, referencia)
    for i, lp in zip(validas, con):
        salida[i] = lp - sin
    return salida


def fluidez(ref_model, tok, prompt_ids, continuacion: str) -> float:
    """Log-prob media por token de la continuación dado el fragmento, sin la imaginación."""
    _, lp = verosimilitud(ref_model, tok, prompt_ids, [], continuacion)
    return lp
