"""Imagina y después continúa un fragmento desde la línea de comandos.

uv run imagine generar "Texto del fragmento…"
uv run imagine generar < fragmento.txt
"""

import argparse
import sys

from imagine.dominio.escritura import ABRE, CIERRA, TOKENS_CONTINUACION
from imagine.ejecucion.escritor import imaginar_y_continuar

from .modelo import cargar
from .opciones import agregar_opciones_del_escritor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "fragmento", nargs="?", help="el texto a continuar; si falta, se lee de la entrada"
    )
    agregar_opciones_del_escritor(ap, largo=TOKENS_CONTINUACION)
    ap.add_argument("--sin-imaginar", action="store_true", help="continuar sin imaginar antes")
    args = ap.parse_args()

    fragmento = args.fragmento if args.fragmento is not None else sys.stdin.read()
    if not fragmento.strip():
        sys.exit("Falta el fragmento: pasalo entre comillas o por la entrada.")
    model, tok = cargar(args.modelo)
    imaginacion, continuacion = imaginar_y_continuar(
        model,
        tok,
        fragmento,
        temperatura=args.temperatura,
        max_tokens=args.max_tokens,
        con_imaginacion=not args.sin_imaginar,
        plantilla=args.plantilla,
    )
    if imaginacion.texto.strip():
        print(f"{ABRE} {imaginacion.texto.strip()} {CIERRA}\n")
    print(continuacion)


if __name__ == "__main__":
    main()
