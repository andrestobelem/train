"""Evalúa en el libro de test (no visto) cuánto ayuda lo imaginado a predecir al autor.

Compará el modelo base contra uno entrenado:

    uv run python -m imagine.evaluar
    uv run python -m imagine.evaluar --adaptadores runs/imagine
"""

import argparse
import json

import mlx.core as mx

from .modelo import cargar, muestrear, prompt, texto_imaginado
from .recompensas import recompensas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="data/test.jsonl")
    ap.add_argument("--adaptadores", default=None)
    ap.add_argument("-n", type=int, default=50, help="fragmentos a evaluar")
    ap.add_argument("--muestras", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--temperatura", type=float, default=0.8)
    args = ap.parse_args()

    mx.random.seed(0)
    model, tok = cargar(args.adaptadores)
    ref_model, _ = cargar()
    ejemplos = [json.loads(l) for l in open(args.datos, encoding="utf-8")][: args.n]

    validas = total = 0
    deltas, rs = [], []
    for i, ej in enumerate(ejemplos, 1):
        p_ids = tok.encode(prompt(ej["contexto"]))
        comps = muestrear(model, tok, p_ids, args.muestras, args.max_tokens, args.temperatura)
        imaginados = [texto_imaginado(tok, c) for c in comps]
        r, d = recompensas(ref_model, tok, p_ids, comps, imaginados, ej["referencia"])
        rs += r
        deltas += [x for x in d if x is not None]
        validas += sum(x is not None for x in d)
        total += len(d)
        if i % 10 == 0:
            print(f"{i}/{len(ejemplos)}…")

    print(f"modelo: {args.adaptadores or 'base'}")
    print(f"formato válido: {validas / total:.1%}")
    print(f"recompensa media: {sum(rs) / len(rs):.3f}")
    if deltas:
        mejora = sum(d > 0 for d in deltas) / len(deltas)
        print(f"Δ log-verosimilitud del autor: {sum(deltas) / len(deltas):+.4f} nats/token")
        print(f"imaginaciones que ayudan (Δ>0): {mejora:.1%}")


if __name__ == "__main__":
    main()
