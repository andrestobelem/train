"""El escritor en acción: imaginar y continuar un fragmento con un modelo cargado."""

from dataclasses import dataclass

from imagine.dominio.escritura import (
    CIERRA,
    FINES_CONTINUACION,
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
) -> tuple[list[int], list[ImaginacionGenerada]]:
    """n imaginaciones del mismo fragmento. Devuelve (ids del prompt, imaginaciones)."""
    p_ids = tok.encode(prompt(fragmento))
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
) -> str:
    """La continuación de la novela después de la imaginación."""
    p_ids = tok.encode(prompt_continuacion(fragmento, imaginacion))
    ids = muestrear(model, tok, p_ids, 1, max_tokens, temperatura, FINES_CONTINUACION)[0]
    return cortar_continuacion(tok.decode(ids))


def imaginar_y_continuar(
    model,
    tok,
    fragmento: str,
    temperatura: float = TEMPERATURA_ESCRITURA,
    max_tokens: int = TOKENS_CONTINUACION,
    con_imaginacion: bool = True,
) -> tuple[Imaginacion, str]:
    """Un turno completo del escritor. Sin imaginar, la imaginación queda vacía."""
    imaginacion = SIN_IMAGINACION
    if con_imaginacion:
        _, (generada,) = imaginar(model, tok, fragmento, 1, temperatura=temperatura)
        imaginacion = generada.imaginacion
    return imaginacion, continuar(model, tok, fragmento, imaginacion, max_tokens, temperatura)


def turno_del_escritor(
    model,
    tok,
    historia: Historia,
    temperatura: float = TEMPERATURA_ESCRITURA,
    max_tokens: int = TOKENS_CONTINUACION,
) -> tuple[Imaginacion, Historia]:
    """El escritor imagina y continúa la historia desde su fragmento final."""
    imaginacion, continuacion = imaginar_y_continuar(
        model, tok, historia.fragmento, temperatura, max_tokens
    )
    return imaginacion, historia.agregar(Turno(Rol.ESCRITOR, continuacion))
