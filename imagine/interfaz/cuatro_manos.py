"""Historia a cuatro manos en la terminal: vos sos el lector y el modelo, el escritor.

uv run imagine cuatro-manos

Escribí un párrafo y apretá Enter; el escritor imagina y sigue la historia. Una línea
vacía pasa el turno. Ctrl-D o /fin terminan.
"""

import argparse
import sys

from imagine.dominio.escritura import ABRE, CIERRA, TOKENS_POR_TURNO
from imagine.dominio.historia import Historia, Rol, Turno
from imagine.ejecucion.escritor import turno_del_escritor

from .modelo import cargar
from .opciones import agregar_opciones_del_escritor

# La imaginación se muestra tenue, sólo si la salida es una terminal.
TENUE, NORMAL = ("\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "")
FIN = "/fin"


def leer_turno_del_lector() -> str | None:
    """El párrafo del lector; None si quiere terminar."""
    try:
        texto = input("\nLector> ")
    except EOFError:
        return None
    return None if texto.strip() == FIN else texto


def main():
    ap = argparse.ArgumentParser()
    agregar_opciones_del_escritor(ap, largo=TOKENS_POR_TURNO)
    ap.add_argument("--ocultar-imaginacion", action="store_true", help="no mostrar lo imaginado")
    args = ap.parse_args()

    model, tok = cargar(args.modelo)
    print(f"Escribí un párrafo. Una línea vacía pasa el turno; {FIN} o Ctrl-D terminan.")
    historia = Historia()
    while (texto := leer_turno_del_lector()) is not None:
        historia = historia.agregar(Turno(Rol.LECTOR, texto))
        imaginacion, historia = turno_del_escritor(
            model,
            tok,
            historia,
            temperatura=args.temperatura,
            max_tokens=args.max_tokens,
            plantilla=args.plantilla,
        )
        if not args.ocultar_imaginacion and imaginacion.texto.strip():
            print(f"\n{TENUE}{ABRE} {imaginacion.texto.strip()} {CIERRA}{NORMAL}")
        print(f"\nEscritor> {historia.turnos[-1].texto}")
    print()


if __name__ == "__main__":
    main()
