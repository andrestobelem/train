"""Cómo se juzga una imaginación: ganancia y recompensa (ver GLOSSARY.md y ADR 0001).

DeepSeek-R1 premia exactitud (¿la respuesta es correcta?) y formato (¿razonó entre
<think>?). Acá no hay una respuesta correcta, pero sabemos qué escribió el autor: la
ganancia mide cuánto ayuda la imaginación a anticipar la referencia.
"""

import math

# Ganancia en nats por token; None cuando la imaginación no tiene formato válido.
Ganancia = float | None


def recompensa(ganancia: Ganancia) -> float:
    """Sin formato válido → 0. Con formato válido → 1 + tanh(4·Δ).

    Una imaginación con formato válido nunca vale menos que una sin formato, y con
    Δ realistas (|Δ| < 1; en la práctica ronda ±0,1) vale estrictamente más. Entre las
    válidas manda cuánto ayudan a predecir al autor.
    """
    return 0.0 if ganancia is None else 1.0 + math.tanh(4 * ganancia)


def ayuda(ganancia: Ganancia) -> bool:
    """Una imaginación ayuda si su ganancia es positiva."""
    return ganancia is not None and ganancia > 0
