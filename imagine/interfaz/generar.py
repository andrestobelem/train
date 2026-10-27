"""Imagina y después continúa un fragmento desde la línea de comandos.

uv run imagine generar "Texto del fragmento…"
uv run imagine generar < fragmento.txt
"""

import argparse
import sys

from imagine.dominio.escritura import TEMPERATURA_ESCRITURA, TOKENS_CONTINUACION
from imagine.ejecucion.escritor import imaginar_y_continuar
from imagine.persistencia import catalogo

from .modelo import cargar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "fragmento", nargs="?", help="el texto a continuar; si falta, se lee de la entrada"
    )
    ap.add_argument(
        "--modelo", default=catalogo.FINAL, help="repo de Hugging Face o carpeta de adaptadores"
    )
    ap.add_argument("--temperatura", type=float, default=TEMPERATURA_ESCRITURA)
    ap.add_argument("--max-tokens", type=int, default=TOKENS_CONTINUACION)
    ap.add_argument("--sin-imaginar", action="store_true", help="continuar sin imaginar antes")
    args = ap.parse_args()

    fragmento = args.fragmento if args.fragmento is not None else sys.stdin.read()
    if not fragmento.strip():
        sys.exit("Falta el fragmento: pasalo entre comillas o por la entrada.")
    model, tok = cargar(args.modelo)
    imaginacion, continuacion = imaginar_y_continuar(
        model, tok, fragmento, args.temperatura, args.max_tokens, not args.sin_imaginar
    )
    if imaginacion.texto.strip():
        print(f"<imagine> {imaginacion.texto.strip()} </imagine>\n")
    print(continuacion)


if __name__ == "__main__":
    main()
