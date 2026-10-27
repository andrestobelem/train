"""La historia a cuatro manos: el lector y el escritor se turnan para escribirla."""

from dataclasses import dataclass, field
from enum import Enum

from .escritura import recortar_fragmento


class Rol(Enum):
    LECTOR = "lector"
    ESCRITOR = "escritor"


@dataclass(frozen=True)
class Turno:
    rol: Rol
    texto: str  # vacío si quien tenía el turno pasó


@dataclass(frozen=True)
class Historia:
    turnos: tuple[Turno, ...] = field(default_factory=tuple)

    @property
    def le_toca(self) -> Rol:
        """Los turnos alternan y empieza el lector."""
        return Rol.LECTOR if len(self.turnos) % 2 == 0 else Rol.ESCRITOR

    def agregar(self, turno: Turno) -> "Historia":
        if turno.rol is not self.le_toca:
            raise ValueError(f"le toca al {self.le_toca.value}, no al {turno.rol.value}")
        return Historia(self.turnos + (turno,))

    @property
    def texto(self) -> str:
        """La historia escrita: un párrafo por turno; los turnos vacíos no agregan nada."""
        return "\n\n".join(t.texto.strip() for t in self.turnos if t.texto.strip())

    @property
    def fragmento(self) -> str:
        """Lo que el escritor ve de la historia en su turno."""
        return recortar_fragmento(self.texto)
