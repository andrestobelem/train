"""El corpus: libros de entrenamiento y de prueba, y los ejemplos que se sacan de ellos."""

from dataclasses import dataclass

from .escritura import Imaginacion, objetivo_supervisado


@dataclass(frozen=True)
class Libro:
    titulo: str
    autor: str

    @property
    def nombre(self) -> str:
        """Cómo se identifica el libro en cada ejemplo: «Título — Autor»."""
        return f"{self.titulo} — {self.autor}"


@dataclass(frozen=True)
class Corpus:
    entrenamiento: tuple[Libro, ...]
    prueba: tuple[Libro, ...]

    def __post_init__(self):
        # La prueba mide si el escritor generaliza a otro autor, no si recuerda uno visto.
        compartidos = {b.autor for b in self.prueba} & {b.autor for b in self.entrenamiento}
        if compartidos:
            raise ValueError(f"autores en entrenamiento y prueba: {sorted(compartidos)}")


@dataclass(frozen=True)
class Ejemplo:
    fragmento: str
    referencia: str  # lo que el autor escribió después; el escritor nunca la ve
    libro: str  # Libro.nombre


@dataclass(frozen=True)
class EjemploSupervisado:
    """Un ejemplo con la imaginación elegida en la selección y su ganancia."""

    ejemplo: Ejemplo
    imaginacion: Imaginacion
    ganancia: float

    @property
    def objetivo(self) -> str:
        return objetivo_supervisado(self.imaginacion, self.ejemplo.referencia)
