"""Imagina y después continúa un fragmento.

    uv run python -m imagine.generar --adaptadores runs/imagine "Texto del fragmento…"
"""

import argparse

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from .modelo import CIERRA, cargar, muestrear, prompt, texto_imaginado


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fragmento")
    ap.add_argument("--adaptadores", default=None)
    ap.add_argument("--temperatura", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    model, tok = cargar(args.adaptadores)
    p = prompt(args.fragmento)
    comp = muestrear(model, tok, tok.encode(p), 1, 128, args.temperatura)[0]
    imaginado, _ = texto_imaginado(tok, comp)
    print(f"<imagine>{imaginado}</imagine>\n")

    continuacion = generate(
        model,
        tok,
        p + imaginado + CIERRA + "\n",
        max_tokens=args.max_tokens,
        sampler=make_sampler(temp=args.temperatura),
    )
    # Se corta donde el modelo empezaría otro turno de la conversación.
    print(continuacion.split("\nLector:")[0].strip())


if __name__ == "__main__":
    main()
