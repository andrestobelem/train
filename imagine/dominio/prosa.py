"""Métricas de la prosa calculables sólo con el texto, sin modelos."""

import unicodedata


def _ngramas(palabras: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(palabras[i : i + n]) for i in range(len(palabras) - n + 1)]


def repeticion(texto: str, n: int = 3) -> float:
    """Fracción de n-gramas de palabras repetidos (menos es mejor)."""
    ngramas = _ngramas(texto.lower().split(), n)
    return 1 - len(set(ngramas)) / len(ngramas) if ngramas else 0.0


def distintos(texto: str, n: int = 2) -> float:
    """Fracción de n-gramas de palabras distintos (más es mejor)."""
    ngramas = _ngramas(texto.lower().split(), n)
    return len(set(ngramas)) / len(ngramas) if ngramas else 0.0


def letras_ajenas(texto: str) -> float:
    """Fracción de letras fuera del alfabeto latino, p. ej. griego o CJK (menos es mejor)."""
    letras = [c for c in texto if c.isalpha()]
    if not letras:
        return 0.0
    return sum(not unicodedata.name(c, "").startswith("LATIN") for c in letras) / len(letras)


def palabras(texto: str) -> int:
    return len(texto.split())
