"""Imagina y después continúa un fragmento desde la línea de comandos.

uv run python -m imagine.interfaz.generar --modelo runs/sft/epoca_1 "Texto del fragmento…"
"""

import argparse

from imagine.dominio.escritura import TEMPERATURA_ESCRITURA, TOKENS_CONTINUACION
from imagine.ejecucion.escritor import imaginar_y_continuar
from imagine.ejecucion.modelos import cargar_por_nombre
from imagine.persistencia import catalogo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fragmento")
    ap.add_argument(
        "--modelo", default=catalogo.FINAL, help="repo de Hugging Face o carpeta de adaptadores"
    )
    ap.add_argument("--temperatura", type=float, default=TEMPERATURA_ESCRITURA)
    ap.add_argument("--max-tokens", type=int, default=TOKENS_CONTINUACION)
    ap.add_argument("--sin-imaginar", action="store_true")
    args = ap.parse_args()

    model, tok = cargar_por_nombre(args.modelo)
    imaginacion, continuacion = imaginar_y_continuar(
        model, tok, args.fragmento, args.temperatura, args.max_tokens, not args.sin_imaginar
    )
    if imaginacion.texto.strip():
        print(f"<imagine> {imaginacion.texto.strip()} </imagine>\n")
    print(continuacion)


if __name__ == "__main__":
    main()
