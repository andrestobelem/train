"""Imagina y después continúa un fragmento.

    uv run python -m imagine.generar --adaptadores runs/sft/epoca_1 "Texto del fragmento…"
"""

import argparse

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from .modelo import ABRE, CIERRA, cargar, muestrear, prompt, texto_imaginado


def imaginar_y_continuar(
    model, tok, fragmento: str, temperatura: float = 0.7, max_tokens: int = 200, imaginar=True
) -> tuple[str, str]:
    """Devuelve (lo imaginado, la continuación). Sin imaginar, la imaginación queda vacía."""
    p = prompt(fragmento)
    imaginado = ""
    if imaginar:
        comp = muestrear(model, tok, tok.encode(p), 1, 128, temperatura)[0]
        imaginado, _ = texto_imaginado(tok, comp)
    continuacion = generate(
        model,
        tok,
        p + imaginado + CIERRA + "\n",
        max_tokens=max_tokens,
        sampler=make_sampler(temp=temperatura),
    )
    # Se corta donde el modelo empezaría otro turno o volvería a las etiquetas.
    for fin in ["\nLector:", ABRE, CIERRA]:
        continuacion = continuacion.split(fin)[0]
    return imaginado.strip(), continuacion.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fragmento")
    ap.add_argument("--adaptadores", default=None)
    ap.add_argument("--temperatura", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    model, tok = cargar(args.adaptadores)
    imaginado, continuacion = imaginar_y_continuar(
        model, tok, args.fragmento, args.temperatura, args.max_tokens
    )
    print(f"<imagine> {imaginado} </imagine>\n")
    print(continuacion)


if __name__ == "__main__":
    main()
