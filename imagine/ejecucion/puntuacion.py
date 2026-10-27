"""Medir contra el corpus con el juez: el modelo base congelado (ver ADR 0001).

Implementa las medidas que necesitan un modelo: la ganancia de cada imaginación y la
fluidez de una continuación. Las reglas que las usan están en `imagine.dominio`.
"""

import mlx.core as mx

from imagine.dominio.escritura import CIERRA, Imaginacion
from imagine.dominio.recompensa import Ganancia

from .lm import logprobs, rellenar


def verosimilitud(juez, tok, prompt_ids, cuerpos_ids, texto: str):
    """Log-prob media por token de `texto` tras cada cuerpo, y tras una imaginación vacía.

    Cada cuerpo es una imaginación más </imagine>. Devuelve (lista por cuerpo, valor vacío).
    """
    cierre = tok.encode(CIERRA, add_special_tokens=False)
    texto_ids = tok.encode("\n" + texto, add_special_tokens=False)
    # La última fila es la línea de base: imaginación vacía.
    cuerpos = [list(c) for c in cuerpos_ids] + [cierre]
    seqs = [prompt_ids + c + texto_ids for c in cuerpos]
    inicios = mx.array([len(prompt_ids) + len(c) for c in cuerpos])
    pos = inicios[:, None] + mx.arange(len(texto_ids))[None]
    lp = logprobs(juez, rellenar(seqs, tok.eos_token_id), pos).mean(axis=1).tolist()
    return lp[:-1], lp[-1]


def ganancias(
    juez, tok, prompt_ids, imaginaciones: list[Imaginacion], referencia: str
) -> list[Ganancia]:
    """La ganancia de cada imaginación sobre la referencia; None si su formato es inválido.

    Se puntúa sólo el texto de la imaginación (más el cierre), re-tokenizado.
    """
    validas = [i for i, im in enumerate(imaginaciones) if im.formato_valido]
    salida: list[Ganancia] = [None] * len(imaginaciones)
    if not validas:
        return salida
    cierre = tok.encode(CIERRA, add_special_tokens=False)
    cuerpos = [
        tok.encode(imaginaciones[i].texto, add_special_tokens=False) + cierre for i in validas
    ]
    con, sin = verosimilitud(juez, tok, prompt_ids, cuerpos, referencia)
    for i, lp in zip(validas, con):
        salida[i] = lp - sin
    return salida


def fluidez(juez, tok, prompt_ids, continuacion: str) -> float:
    """Log-prob media por token de la continuación dado el fragmento, sin la imaginación."""
    _, lp = verosimilitud(juez, tok, prompt_ids, [], continuacion)
    return lp
