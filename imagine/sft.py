"""Rejection sampling + SFT, la etapa que sigue al RL en DeepSeek-R1.

1. muestrear: por cada fragmento de entrenamiento, el modelo de RL imagina G veces y
   nos quedamos con la imaginación que más ayuda a predecir al autor (Δ > 0),
   verificada contra el corpus igual que en la recompensa.
2. entrenar: ajuste supervisado sobre fragmento → mejor imaginación → texto real del
   autor. Así el modelo aprende a escribir como el autor después de imaginar, que el
   RL nunca le enseñó: sólo premiaba la imaginación.

    uv run python -m imagine.sft muestrear --adaptadores runs/imagine300/paso_00250
    uv run python -m imagine.sft entrenar --adaptadores runs/imagine300/paso_00250
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from .modelo import (
    CIERRA,
    cargar,
    logprobs_completacion,
    muestrear,
    prellenar,
    prompt,
    texto_imaginado,
)
from .recompensas import recompensas

# Marca de fin: el próximo turno de la conversación. Enseña dónde parar.
FIN = "\n\nLector:"


def cmd_muestrear(args):
    random.seed(args.semilla)
    mx.random.seed(args.semilla)
    ejemplos = [json.loads(l) for l in open(args.datos, encoding="utf-8")]
    ejemplos = random.sample(ejemplos, args.n)
    salida = Path(args.salida_datos)
    hechos = sum(1 for _ in open(salida)) if salida.exists() else 0
    procesados = Path(str(salida) + ".progreso")
    inicio = int(procesados.read_text()) if procesados.exists() else 0
    print(f"retomando desde el fragmento {inicio} ({hechos} ejemplos guardados)")

    model, tok = cargar(args.adaptadores)
    ref_model, _ = cargar()
    with open(salida, "a", encoding="utf-8") as f:
        for i in range(inicio, len(ejemplos)):
            ej = ejemplos[i]
            t0 = time.time()
            p_ids = tok.encode(prompt(ej["contexto"]))
            comps = muestrear(model, tok, p_ids, args.grupo, args.max_tokens, args.temperatura)
            imaginados = [texto_imaginado(tok, c) for c in comps]
            _, deltas = recompensas(ref_model, tok, p_ids, comps, imaginados, ej["referencia"])
            validos = [(d, j) for j, d in enumerate(deltas) if d is not None and d > 0]
            if validos:
                d, j = max(validos)
                fila = {**ej, "imaginado": imaginados[j][0], "delta": d}
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
                f.flush()
                hechos += 1
            procesados.write_text(str(i + 1))
            mejor = f"{max(validos)[0]:+.4f}" if validos else "  —   "
            print(f"{i + 1}/{len(ejemplos)} | mejor Δ {mejor} | guardados {hechos} | "
                  f"{time.time() - t0:.1f}s")


def perdida_sft(model, cache, ultimo, comps):
    return -logprobs_completacion(model, cache, ultimo, comps).mean()


def preparar(tok, fila):
    p_ids = tok.encode(prompt(fila["contexto"]))
    texto = fila["imaginado"] + CIERRA + "\n" + fila["referencia"] + FIN
    return p_ids, tok.encode(texto, add_special_tokens=False)


def guardar(model, origen: Path, destino: Path):
    destino.mkdir(parents=True, exist_ok=True)
    # Misma configuración de LoRA que los adaptadores de partida.
    shutil.copy(origen / "adapter_config.json", destino / "adapter_config.json")
    pesos = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(destino / "adapters.safetensors"), pesos)


def cmd_entrenar(args):
    random.seed(args.semilla)
    filas = [json.loads(l) for l in open(args.salida_datos, encoding="utf-8")]
    random.shuffle(filas)
    n_val = max(1, len(filas) // 20)
    val, train = filas[:n_val], filas[n_val:]
    print(f"train {len(train)} | val {len(val)}")

    model, tok = cargar(args.adaptadores)
    # Sólo se entrenan las matrices de LoRA, partiendo de las del RL.
    model.freeze()
    model.apply_to_modules(
        lambda _, m: m.unfreeze(keys=["lora_a", "lora_b"], strict=False)
        if hasattr(m, "lora_a")
        else None
    )
    n = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"parámetros entrenables: {n / 1e6:.1f}M")
    opt = optim.AdamW(learning_rate=args.lr, weight_decay=0.0)
    grad_fn = nn.value_and_grad(model, perdida_sft)

    def evaluar():
        total = 0.0
        for fila in val:
            p_ids, c_ids = preparar(tok, fila)
            cache = prellenar(model, p_ids, 1)
            total += perdida_sft(model, cache, p_ids[-1], mx.array([c_ids])).item()
        return total / len(val)

    salida = Path(args.salida)
    log = open(salida.parent / f"{salida.name}_log.jsonl", "a", encoding="utf-8")
    print(f"época 0 | val {evaluar():.4f}")
    paso = 0
    for epoca in range(1, args.epocas + 1):
        random.shuffle(train)
        acum, perdidas, t0 = None, [], time.time()
        for k, fila in enumerate(train, 1):
            p_ids, c_ids = preparar(tok, fila)
            cache = prellenar(model, p_ids, 1)
            model.train()
            loss, g = grad_fn(model, cache, p_ids[-1], mx.array([c_ids]))
            model.eval()
            acum = g if acum is None else tree_map(mx.add, acum, g)
            mx.eval(acum, loss)
            perdidas.append(loss.item())
            if k % args.acumular == 0 or k == len(train):
                opt.update(model, tree_map(lambda x: x / args.acumular, acum))
                mx.eval(model.trainable_parameters(), opt.state)
                acum = None
                paso += 1
            if k % 50 == 0:
                media = sum(perdidas[-50:]) / len(perdidas[-50:])
                print(f"época {epoca} | {k}/{len(train)} | loss {media:.4f} | "
                      f"{time.time() - t0:.0f}s")
        v = evaluar()
        fila_log = {"epoca": epoca, "loss": sum(perdidas) / len(perdidas), "val": v}
        log.write(json.dumps(fila_log) + "\n")
        log.flush()
        print(f"época {epoca} | train {fila_log['loss']:.4f} | val {v:.4f}")
        guardar(model, Path(args.adaptadores), salida / f"epoca_{epoca}")
    guardar(model, Path(args.adaptadores), salida)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ["muestrear", "entrenar"]:
        p = sub.add_parser(nombre)
        p.add_argument("--adaptadores", default="runs/imagine300/paso_00250",
                       help="modelo de RL del que se parte")
        p.add_argument("--salida-datos", default="data/sft.jsonl")
        p.add_argument("--semilla", type=int, default=1)
    m = sub.choices["muestrear"]
    m.add_argument("--datos", default="data/train.jsonl")
    m.add_argument("-n", type=int, default=600, help="fragmentos a muestrear")
    m.add_argument("--grupo", type=int, default=8)
    m.add_argument("--max-tokens", type=int, default=96)
    m.add_argument("--temperatura", type=float, default=0.8)
    e = sub.choices["entrenar"]
    e.add_argument("--salida", default="runs/sft")
    e.add_argument("--epocas", type=int, default=2)
    e.add_argument("--lr", type=float, default=5e-6)
    e.add_argument("--acumular", type=int, default=4, help="ejemplos por actualización")
    args = ap.parse_args()
    {"muestrear": cmd_muestrear, "entrenar": cmd_entrenar}[args.cmd](args)


if __name__ == "__main__":
    main()
