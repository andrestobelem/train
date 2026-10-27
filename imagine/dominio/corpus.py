"""Ejemplos del corpus: un fragmento de un libro y la referencia que escribió su autor."""

from dataclasses import dataclass

from .escritura import Imaginacion, objetivo_supervisado


@dataclass(frozen=True)
class Ejemplo:
    fragmento: str
    referencia: str  # lo que el autor escribió después; el escritor nunca la ve
    libro: str


@dataclass(frozen=True)
class EjemploSupervisado:
    """Un ejemplo con la imaginación elegida en la selección y su ganancia."""

    ejemplo: Ejemplo
    imaginacion: Imaginacion
    ganancia: float

    @property
    def objetivo(self) -> str:
        return objetivo_supervisado(self.imaginacion, self.ejemplo.referencia)
