"""La selección: por cada fragmento, la imaginación de mayor ganancia, si ayuda."""

from .escritura import Imaginacion
from .recompensa import Ganancia, ayuda


def seleccionar(
    imaginaciones: list[Imaginacion], ganancias: list[Ganancia]
) -> tuple[Imaginacion, float] | None:
    """La imaginación elegida y su ganancia; None si ninguna ayuda (el fragmento se descarta)."""
    candidatas = [(g, i) for i, g in enumerate(ganancias) if ayuda(g)]
    if not candidatas:
        return None
    ganancia, i = max(candidatas)
    return imaginaciones[i], ganancia


def mejor_imaginacion(
    imaginaciones: list[Imaginacion], ganancias: list[Ganancia]
) -> Imaginacion | None:
    """La de mayor ganancia entre las de formato válido; None si ninguna es válida.

    Coincide con la de mayor recompensa, porque la recompensa crece con la ganancia.
    """
    validas = [i for i, g in enumerate(ganancias) if g is not None]
    if not validas:
        return None
    return imaginaciones[max(validas, key=lambda i: ganancias[i])]
