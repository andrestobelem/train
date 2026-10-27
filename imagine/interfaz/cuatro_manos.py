"""Historia a cuatro manos en la terminal: vos sos el lector y el modelo, el escritor.

uv run imagine cuatro-manos

Escribí un párrafo y apretá Enter; el escritor imagina y sigue la historia. Una línea
vacía pasa el turno. Ctrl-D o /fin terminan.
"""

import argparse
import sys

from imagine.dominio.escritura import TEMPERATURA_ESCRITURA
from imagine.dominio.historia import Historia, Rol, Turno
from imagine.ejecucion.escritor import turno_del_escritor
from imagine.persistencia import catalogo

from .modelo import cargar

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
    ap.add_argument(
        "--modelo", default=catalogo.FINAL, help="repo de Hugging Face o carpeta de adaptadores"
    )
    ap.add_argument("--temperatura", type=float, default=TEMPERATURA_ESCRITURA)
    ap.add_argument("--max-tokens", type=int, default=160, help="largo máximo de cada turno")
    ap.add_argument("--ocultar-imaginacion", action="store_true", help="no mostrar lo imaginado")
    args = ap.parse_args()

    model, tok = cargar(args.modelo)
    print(f"Escribí un párrafo. Una línea vacía pasa el turno; {FIN} o Ctrl-D terminan.")
    historia = Historia()
    while (texto := leer_turno_del_lector()) is not None:
        historia = historia.agregar(Turno(Rol.LECTOR, texto))
        imaginacion, historia = turno_del_escritor(
            model, tok, historia, args.temperatura, args.max_tokens
        )
        if not args.ocultar_imaginacion and imaginacion.texto.strip():
            print(f"\n{TENUE}<imagine> {imaginacion.texto.strip()} </imagine>{NORMAL}")
        print(f"\nEscritor> {historia.turnos[-1].texto}")
    print()


if __name__ == "__main__":
    main()
