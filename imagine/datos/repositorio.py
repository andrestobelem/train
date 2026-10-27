"""Repositorio: guarda y recupera ejemplos del dominio en archivos jsonl.

El formato en disco es anterior al glosario y se conserva para no invalidar los datos
ya armados: el fragmento se guarda como "contexto", la imaginación elegida como
"imaginado" y su ganancia como "delta".
"""

from pathlib import Path
from typing import IO

from imagine.dominio.corpus import Ejemplo, EjemploSupervisado
from imagine.dominio.escritura import Imaginacion

from .archivos import SFT, TEST, TRAIN, agregar_jsonl, escribir_jsonl, leer_jsonl


def _ejemplo(fila: dict) -> Ejemplo:
    return Ejemplo(fila["contexto"], fila["referencia"], fila["libro"])


def _fila(ejemplo: Ejemplo) -> dict:
    return {"libro": ejemplo.libro, "contexto": ejemplo.fragmento, "referencia": ejemplo.referencia}


def leer_ejemplos(ruta: str | Path, n: int | None = None) -> list[Ejemplo]:
    return [_ejemplo(f) for f in leer_jsonl(ruta, n)]


def escribir_ejemplos(ruta: str | Path, ejemplos: list[Ejemplo]) -> None:
    escribir_jsonl(ruta, [_fila(e) for e in ejemplos])


def ejemplos_de_entrenamiento() -> list[Ejemplo]:
    return leer_ejemplos(TRAIN)


def ejemplos_de_prueba(n: int | None = None) -> list[Ejemplo]:
    """Ejemplos del libro de prueba (Niebla)."""
    return leer_ejemplos(TEST, n)


def leer_supervisados(ruta: str | Path = SFT) -> list[EjemploSupervisado]:
    return [
        EjemploSupervisado(_ejemplo(f), Imaginacion(f["imaginado"]), f["delta"])
        for f in leer_jsonl(ruta)
    ]


def agregar_supervisado(archivo: IO[str], supervisado: EjemploSupervisado) -> None:
    fila = _fila(supervisado.ejemplo)
    fila.update(imaginado=supervisado.imaginacion.texto, delta=supervisado.ganancia)
    agregar_jsonl(archivo, fila)
