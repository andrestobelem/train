"""La recompensa, al estilo DeepSeek-R1: basada en reglas y verificable contra el corpus.

R1 premia exactitud (¿la respuesta es correcta?) y formato (¿razonó entre <think>?).
Acá no hay una respuesta correcta, pero sabemos qué escribió el autor a continuación:

- Formato: ver `escritura.formato_ok`.
- Corpus: Δ = cuánto mejora lo imaginado la log-verosimilitud media por token del
  texto real del autor, frente a una imaginación vacía. La imaginación no ve la
  referencia: sólo gana si anticipa lo que de verdad pasa en la novela.
"""

import math


def recompensa(delta: float | None) -> float:
    """Sin formato válido (Δ = None) → 0. Válida → 1 + tanh(4·Δ), con Δ en nats/token.

    Una imaginación con formato correcto nunca vale menos que una sin formato, y con
    Δ realistas (|Δ| < 1; en la práctica ronda ±0,1) vale estrictamente más. Entre las
    válidas manda cuánto ayudan a predecir al autor.
    """
    return 0.0 if delta is None else 1.0 + math.tanh(4 * delta)
