"""El escritor en acción: imaginar y continuar un fragmento con un modelo cargado."""

from dataclasses import dataclass

from imagine.dominio.escritura import (
    CIERRA,
    SIN_IMAGINACION,
    TEMPERATURA_ESCRITURA,
    TEMPERATURA_MUESTREO,
    TOKENS_CONTINUACION,
    TOKENS_IMAGINACION,
    Imaginacion,
    cortar_continuacion,
    prompt,
    prompt_continuacion,
)
from imagine.dominio.historia import Historia, Rol, Turno
from imagine.dominio.plantillas import POR_DEFECTO, Plantilla

from .lm import muestrear


@dataclass(frozen=True)
class ImaginacionGenerada:
    """Una imaginación junto con los tokens exactos que la produjeron (para entrenar)."""

    ids: list[int]
    imaginacion: Imaginacion


def imaginar(
    model,
    tok,
    fragmento: str,
    n: int = 1,
    max_tokens: int = TOKENS_IMAGINACION,
    temperatura: float = TEMPERATURA_MUESTREO,
    plantilla: Plantilla = POR_DEFECTO,
) -> tuple[list[int], list[ImaginacionGenerada]]:
    """n imaginaciones del mismo fragmento. Devuelve (ids del prompt, imaginaciones)."""
    p_ids = tok.encode(prompt(fragmento, plantilla))
    salidas = muestrear(model, tok, p_ids, n, max_tokens, temperatura, (CIERRA,))
    generadas = [
        ImaginacionGenerada(ids, Imaginacion.desde_generacion(tok.decode(ids))) for ids in salidas
    ]
    return p_ids, generadas


def continuar(
    model,
    tok,
    fragmento: str,
    imaginacion: Imaginacion,
    max_tokens: int = TOKENS_CONTINUACION,
    temperatura: float = TEMPERATURA_ESCRITURA,
    plantilla: Plantilla = POR_DEFECTO,
) -> str:
    """La continuación de la novela después de la imaginación."""
    p_ids = tok.encode(prompt_continuacion(fragmento, imaginacion, plantilla))
    ids = muestrear(model, tok, p_ids, 1, max_tokens, temperatura, plantilla.fines)[0]
    return cortar_continuacion(tok.decode(ids), plantilla)


def imaginar_y_continuar(
    model,
    tok,
    fragmento: str,
    temperatura: float = TEMPERATURA_ESCRITURA,
    max_tokens: int = TOKENS_CONTINUACION,
    con_imaginacion: bool = True,
    plantilla: Plantilla = POR_DEFECTO,
) -> tuple[Imaginacion, str]:
    """Un turno completo del escritor. Sin imaginar, la imaginación queda vacía."""
    imaginacion = SIN_IMAGINACION
    if con_imaginacion:
        _, (generada,) = imaginar(
            model, tok, fragmento, 1, temperatura=temperatura, plantilla=plantilla
        )
        imaginacion = generada.imaginacion
    continuacion = continuar(model, tok, fragmento, imaginacion, max_tokens, temperatura, plantilla)
    return imaginacion, continuacion


def turno_del_escritor(
    model,
    tok,
    historia: Historia,
    temperatura: float = TEMPERATURA_ESCRITURA,
    max_tokens: int = TOKENS_CONTINUACION,
    plantilla: Plantilla = POR_DEFECTO,
) -> tuple[Imaginacion, Historia]:
    """El escritor imagina y continúa la historia desde su fragmento final."""
    imaginacion, continuacion = imaginar_y_continuar(
        model, tok, historia.fragmento, temperatura, max_tokens, plantilla=plantilla
    )
    return imaginacion, historia.agregar(Turno(Rol.ESCRITOR, continuacion))
