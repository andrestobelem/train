"""El escritor: imaginar y continuar un fragmento con un modelo cargado."""

from dataclasses import dataclass

from imagine.dominio.escritura import (
    CIERRA,
    FINES_CONTINUACION,
    TEMPERATURA_ESCRITURA,
    TEMPERATURA_MUESTREO,
    TOKENS_CONTINUACION,
    TOKENS_IMAGINACION,
    cortar_continuacion,
    prompt,
    prompt_continuacion,
    separar_imaginado,
)

from .lm import muestrear


@dataclass
class Imaginacion:
    ids: list[int]  # los tokens generados, tal cual (para entrenar sobre ellos)
    texto: str  # lo imaginado, sin la etiqueta de cierre
    cerro: bool  # si cerró </imagine>


def imaginar(
    model,
    tok,
    fragmento: str,
    n: int = 1,
    max_tokens: int = TOKENS_IMAGINACION,
    temperatura: float = TEMPERATURA_MUESTREO,
) -> tuple[list[int], list[Imaginacion]]:
    """n imaginaciones del mismo fragmento. Devuelve (ids del prompt, imaginaciones)."""
    p_ids = tok.encode(prompt(fragmento))
    salidas = muestrear(model, tok, p_ids, n, max_tokens, temperatura, (CIERRA,))
    return p_ids, [Imaginacion(ids, *separar_imaginado(tok.decode(ids))) for ids in salidas]


def continuar(
    model,
    tok,
    fragmento: str,
    imaginado: str,
    max_tokens: int = TOKENS_CONTINUACION,
    temperatura: float = TEMPERATURA_ESCRITURA,
) -> str:
    """La continuación de la novela después de lo imaginado."""
    p_ids = tok.encode(prompt_continuacion(fragmento, imaginado))
    ids = muestrear(model, tok, p_ids, 1, max_tokens, temperatura, FINES_CONTINUACION)[0]
    return cortar_continuacion(tok.decode(ids))


def imaginar_y_continuar(
    model,
    tok,
    fragmento: str,
    temperatura: float = TEMPERATURA_ESCRITURA,
    max_tokens: int = TOKENS_CONTINUACION,
    con_imaginacion: bool = True,
) -> tuple[str, str]:
    """Devuelve (lo imaginado, la continuación). Sin imaginar, la imaginación queda vacía."""
    imaginado = ""
    if con_imaginacion:
        _, (imaginacion,) = imaginar(model, tok, fragmento, 1, temperatura=temperatura)
        imaginado = imaginacion.texto
    return imaginado.strip(), continuar(model, tok, fragmento, imaginado, max_tokens, temperatura)
