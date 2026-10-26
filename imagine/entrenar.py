"""GRPO (Group Relative Policy Optimization) sobre un modelo base, como en DeepSeek-R1-Zero.

Por cada fragmento se muestrean G imaginaciones, se puntúan contra el corpus y la
ventaja de cada una es su recompensa normalizada dentro del grupo (sin crítico):

    A_i = (r_i − media(r)) / std(r)

La pérdida es el objetivo recortado de PPO más una penalización KL (estimador k3)
contra el modelo de referencia congelado. Se entrena con LoRA para que entre
cómodo en una Mac.
"""

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map
from mlx_lm.tuner.utils import linear_to_lora_layers

from .modelo import (
    MODELO_BASE,
    cargar,
    logprobs_completacion,
    muestrear,
    prellenar,
    prompt,
    rellenar,
    texto_imaginado,
)
from .recompensas import recompensas


def perdida_grpo(model, cache, ultimo, comps, mascara, ventajas, ref_lp, viejo_lp, args):
    lp = logprobs_completacion(model, cache, ultimo, comps)
    ratio = mx.exp(lp - viejo_lp)
    pg = -mx.minimum(ratio * ventajas, mx.clip(ratio, 1 - args.eps, 1 + args.eps) * ventajas)
    kl = mx.exp(ref_lp - lp) - (ref_lp - lp) - 1
    por_token = (pg + args.beta * kl) * mascara
    # Como en el paper: promedio por completación (1/|o_i|) y después por grupo. Se
    # divide por G (no por el micro-lote) para que acumular micro-lotes dé la media.
    por_seq = por_token.sum(axis=1) / mascara.sum(axis=1)
    return por_seq.sum() / args.grupo, (kl * mascara).sum()


def paso_grupo(model, ref_model, tok, ejemplo, args):
    """Muestrea un grupo, calcula recompensas y devuelve (grads, métricas)."""
    p_ids = tok.encode(prompt(ejemplo["contexto"]))
    comps = muestrear(model, tok, p_ids, args.grupo, args.max_tokens, args.temperatura)
    imaginados = [texto_imaginado(tok, c) for c in comps]
    r, deltas = recompensas(ref_model, tok, p_ids, comps, imaginados, ejemplo["referencia"])

    r_arr = mx.array(r)
    std = r_arr.std().item()
    metricas = {
        "recompensa": sum(r) / len(r),
        "formato": sum(d is not None for d in deltas) / len(deltas),
        "delta": [d for d in deltas if d is not None],
        "largo": sum(len(c) for c in comps) / len(comps),
        "muestra": imaginados[max(range(len(r)), key=r.__getitem__)][0],
    }
    if std < 1e-6:  # todas iguales: ventaja cero, no hay señal
        return None, metricas
    ventajas = ((r_arr - r_arr.mean()) / (std + 1e-4))[:, None]

    grad_fn = nn.value_and_grad(model, perdida_grpo)
    grads, loss, kl, n_tokens = None, 0.0, 0.0, 0
    for i in range(0, len(comps), args.micro_lote):
        sub = comps[i : i + args.micro_lote]
        ids = rellenar(sub, tok.eos_token_id)
        mascara = (mx.arange(ids.shape[1])[None] < mx.array([len(c) for c in sub])[:, None])
        mascara = mascara.astype(mx.float32)
        n = len(sub)
        ref_lp = logprobs_completacion(ref_model, prellenar(ref_model, p_ids, n), p_ids[-1], ids)
        # Una sola actualización por lote (on-policy): π_old = π_θ sin gradiente.
        viejo_lp = logprobs_completacion(model, prellenar(model, p_ids, n), p_ids[-1], ids)
        mx.eval(ref_lp, viejo_lp)

        cache = prellenar(model, p_ids, n)
        # En modo train la atención lineal usa ops diferenciables en vez del kernel Metal.
        model.train()
        (l, k), g = grad_fn(
            model, cache, p_ids[-1], ids, mascara, ventajas[i : i + n], ref_lp, viejo_lp, args
        )
        model.eval()
        grads = g if grads is None else tree_map(mx.add, grads, g)
        mx.eval(grads, l, k)
        loss += l.item()
        kl += k.item()
        n_tokens += mascara.sum().item()
    metricas.update(loss=loss, kl=kl / n_tokens)
    return grads, metricas


def guardar(model, salida: Path, args):
    salida.mkdir(parents=True, exist_ok=True)
    pesos = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(salida / "adapters.safetensors"), pesos)
    config = {
        "fine_tune_type": "lora",
        "num_layers": args.capas,
        "lora_parameters": {"rank": args.rango, "scale": args.escala, "dropout": 0.0},
        "model": MODELO_BASE,
    }
    (salida / "adapter_config.json").write_text(json.dumps(config, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", default="data/train.jsonl")
    ap.add_argument("--salida", default="runs/imagine")
    ap.add_argument("--pasos", type=int, default=300)
    ap.add_argument("--prompts-por-paso", type=int, default=4)
    ap.add_argument("--grupo", type=int, default=8, help="G: muestras por prompt")
    ap.add_argument("--micro-lote", type=int, default=2, help="completaciones por backward")
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--temperatura", type=float, default=0.8)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.04, help="peso de la KL")
    ap.add_argument("--eps", type=float, default=0.2, help="recorte de PPO")
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--escala", type=float, default=20.0)
    ap.add_argument("--capas", type=int, default=-1, help="capas con LoRA (-1 = todas)")
    ap.add_argument("--guardar-cada", type=int, default=50)
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.semilla)
    mx.random.seed(args.semilla)
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    ejemplos = [json.loads(l) for l in open(args.datos, encoding="utf-8")]
    model, tok = cargar()
    ref_model, _ = cargar()
    ref_model.freeze()
    model.freeze()
    linear_to_lora_layers(
        model, args.capas, {"rank": args.rango, "scale": args.escala, "dropout": 0.0}
    )
    n_entrenables = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"parámetros LoRA: {n_entrenables / 1e6:.1f}M")
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)

    log = open(salida / "log.jsonl", "a", encoding="utf-8")
    for paso in range(1, args.pasos + 1):
        t0 = time.time()
        acum, n_grads, ms = None, 0, []
        for ej in random.sample(ejemplos, args.prompts_por_paso):
            grads, m = paso_grupo(model, ref_model, tok, ej, args)
            ms.append(m)
            if grads is None:
                continue
            acum = grads if acum is None else tree_map(mx.add, acum, grads)
            n_grads += 1
        if acum is not None:
            acum = tree_map(lambda g: g / n_grads, acum)
            opt.update(model, acum)
            mx.eval(model.trainable_parameters(), opt.state)

        deltas = [d for m in ms for d in m["delta"]]
        fila = {
            "paso": paso,
            "recompensa": sum(m["recompensa"] for m in ms) / len(ms),
            "formato": sum(m["formato"] for m in ms) / len(ms),
            "delta": sum(deltas) / len(deltas) if deltas else None,
            "largo": sum(m["largo"] for m in ms) / len(ms),
            "kl": sum(m.get("kl", 0) for m in ms) / len(ms),
            "grupos_con_señal": n_grads,
            "seg": round(time.time() - t0, 1),
            "mem_gb": round(mx.get_peak_memory() / 1e9, 1),
            "muestra": ms[0]["muestra"],
        }
        log.write(json.dumps(fila, ensure_ascii=False) + "\n")
        log.flush()
        d = f"{fila['delta']:+.4f}" if fila["delta"] is not None else "  —   "
        print(
            f"paso {paso:4d} | r {fila['recompensa']:.3f} | formato {fila['formato']:.2f} | "
            f"Δ {d} | largo {fila['largo']:.0f} | kl {fila['kl']:.4f} | {fila['seg']}s | {fila['mem_gb']}GB"
        )
        if paso % args.guardar_cada == 0 or paso == args.pasos:
            guardar(model, salida, args)
            # Cada checkpoint en su carpeta, cargable con --adaptadores.
            guardar(model, salida / f"paso_{paso:05d}", args)


if __name__ == "__main__":
    main()
