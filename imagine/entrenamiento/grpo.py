"""GRPO (Group Relative Policy Optimization) sobre un modelo base, como en DeepSeek-R1-Zero.

Por cada fragmento se muestrean G imaginaciones, se puntúan contra el corpus y la
ventaja de cada una es su recompensa normalizada dentro del grupo (sin crítico):

    A_i = (r_i − media(r)) / std(r)

La pérdida es el objetivo recortado de PPO más una penalización KL (estimador k3)
contra el modelo de referencia congelado. Se entrena con LoRA para que entre
cómodo en una Mac.

    uv run python -m imagine.entrenamiento.grpo --salida runs/imagine300
"""

import argparse
import random
import time
from dataclasses import dataclass, fields

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from imagine.datos.archivos import TRAIN, agregar_jsonl, leer_jsonl, ruta_del_proyecto
from imagine.dominio.escritura import TEMPERATURA_MUESTREO, TOKENS_IMAGINACION
from imagine.dominio.recompensa import recompensa
from imagine.ejecucion import escritor, puntuacion
from imagine.ejecucion.lm import logprobs_completacion, prellenar, rellenar
from imagine.ejecucion.modelos import MODELO_BASE, cargar
from imagine.persistencia import adaptadores, catalogo

from .lora import agregar_lora, millones_entrenables


@dataclass
class ConfigGRPO:
    datos: str = str(TRAIN)
    salida: str = catalogo.SALIDA_GRPO
    pasos: int = 300
    prompts_por_paso: int = 4
    grupo: int = 8  # G: muestras por prompt
    micro_lote: int = 2  # completaciones por backward
    max_tokens: int = TOKENS_IMAGINACION
    temperatura: float = TEMPERATURA_MUESTREO
    lr: float = 2e-6
    beta: float = 0.04  # peso de la KL
    eps: float = 0.2  # recorte de PPO
    rango: int = 16
    escala: float = 20.0
    capas: int = -1  # capas con LoRA (-1 = todas)
    guardar_cada: int = 50
    semilla: int = 0


def perdida_grpo(model, cache, ultimo, comps, mascara, ventajas, ref_lp, viejo_lp, eps, beta, g):
    lp = logprobs_completacion(model, cache, ultimo, comps)
    ratio = mx.exp(lp - viejo_lp)
    pg = -mx.minimum(ratio * ventajas, mx.clip(ratio, 1 - eps, 1 + eps) * ventajas)
    kl = mx.exp(ref_lp - lp) - (ref_lp - lp) - 1
    por_token = (pg + beta * kl) * mascara
    # Como en el paper: promedio por completación (1/|o_i|) y después por grupo. Se
    # divide por G (no por el micro-lote) para que acumular micro-lotes dé la media.
    por_seq = por_token.sum(axis=1) / mascara.sum(axis=1)
    return por_seq.sum() / g, (kl * mascara).sum()


def ventajas_de_grupo(recompensas: list[float]) -> mx.array | None:
    """A_i normalizadas en el grupo; None si son todas iguales (no hay señal)."""
    r = mx.array(recompensas)
    std = r.std().item()
    if std < 1e-6:
        return None
    return ((r - r.mean()) / (std + 1e-4))[:, None]


def paso_grupo(model, ref_model, tok, ejemplo: dict, conf: ConfigGRPO):
    """Muestrea un grupo, calcula recompensas y devuelve (grads o None, métricas)."""
    p_ids, imaginaciones = escritor.imaginar(
        model, tok, ejemplo["contexto"], conf.grupo, conf.max_tokens, conf.temperatura
    )
    comps = [im.ids for im in imaginaciones]
    d = puntuacion.deltas(
        ref_model, tok, p_ids, [(im.texto, im.cerro) for im in imaginaciones], ejemplo["referencia"]
    )
    r = [recompensa(x) for x in d]
    metricas = {
        "recompensa": sum(r) / len(r),
        "formato": sum(x is not None for x in d) / len(d),
        "delta": [x for x in d if x is not None],
        "largo": sum(len(c) for c in comps) / len(comps),
        "muestra": imaginaciones[max(range(len(r)), key=r.__getitem__)].texto,
    }
    ventajas = ventajas_de_grupo(r)
    if ventajas is None:
        return None, metricas

    grad_fn = nn.value_and_grad(model, perdida_grpo)
    grads, loss, kl, n_tokens = None, 0.0, 0.0, 0
    for i in range(0, len(comps), conf.micro_lote):
        sub = comps[i : i + conf.micro_lote]
        ids = rellenar(sub, tok.eos_token_id)
        mascara = mx.arange(ids.shape[1])[None] < mx.array([len(c) for c in sub])[:, None]
        mascara = mascara.astype(mx.float32)
        n = len(sub)
        ref_lp = logprobs_completacion(ref_model, prellenar(ref_model, p_ids, n), p_ids[-1], ids)
        # Una sola actualización por lote (on-policy): π_old = π_θ sin gradiente.
        viejo_lp = logprobs_completacion(model, prellenar(model, p_ids, n), p_ids[-1], ids)
        mx.eval(ref_lp, viejo_lp)

        cache = prellenar(model, p_ids, n)
        # En modo train la atención lineal usa ops diferenciables en vez del kernel Metal.
        model.train()
        (perdida, k), g = grad_fn(
            model,
            cache,
            p_ids[-1],
            ids,
            mascara,
            ventajas[i : i + n],
            ref_lp,
            viejo_lp,
            conf.eps,
            conf.beta,
            conf.grupo,
        )
        model.eval()
        grads = g if grads is None else tree_map(mx.add, grads, g)
        mx.eval(grads, perdida, k)
        loss += perdida.item()
        kl += k.item()
        n_tokens += mascara.sum().item()
    metricas.update(loss=loss, kl=kl / n_tokens)
    return grads, metricas


def entrenar(conf: ConfigGRPO) -> None:
    # Generador propio: el estado global de random lo tocan librerías al importarse.
    rng = random.Random(conf.semilla)
    mx.random.seed(conf.semilla)
    salida = ruta_del_proyecto(conf.salida)
    salida.mkdir(parents=True, exist_ok=True)
    config_lora = adaptadores.config_lora(MODELO_BASE, conf.capas, conf.rango, conf.escala)

    ejemplos = leer_jsonl(conf.datos)
    model, tok = cargar()
    ref_model, _ = cargar()
    ref_model.freeze()
    agregar_lora(model, conf.capas, conf.rango, conf.escala)
    print(f"parámetros LoRA: {millones_entrenables(model):.1f}M")
    opt = optim.AdamW(learning_rate=conf.lr, weight_decay=0.0)

    log = open(salida / "log.jsonl", "a", encoding="utf-8")
    for paso in range(1, conf.pasos + 1):
        t0 = time.time()
        acum, n_grads, ms = None, 0, []
        for ej in rng.sample(ejemplos, conf.prompts_por_paso):
            grads, m = paso_grupo(model, ref_model, tok, ej, conf)
            ms.append(m)
            if grads is None:
                continue
            acum = grads if acum is None else tree_map(mx.add, acum, grads)
            n_grads += 1
        if acum is not None:
            opt.update(model, tree_map(lambda g: g / n_grads, acum))
            mx.eval(model.trainable_parameters(), opt.state)

        deltas = [x for m in ms for x in m["delta"]]
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
        agregar_jsonl(log, fila)
        d = f"{fila['delta']:+.4f}" if fila["delta"] is not None else "  —   "
        print(
            f"paso {paso:4d} | r {fila['recompensa']:.3f} | formato {fila['formato']:.2f} | "
            f"Δ {d} | largo {fila['largo']:.0f} | kl {fila['kl']:.4f} | "
            f"{fila['seg']}s | {fila['mem_gb']}GB"
        )
        if paso % conf.guardar_cada == 0 or paso == conf.pasos:
            adaptadores.guardar(model, salida, config_lora)
            # Cada checkpoint en su carpeta, cargable como modelo.
            adaptadores.guardar(model, salida / f"paso_{paso:05d}", config_lora)


def main():
    ap = argparse.ArgumentParser()
    for campo in fields(ConfigGRPO):
        ap.add_argument(
            "--" + campo.name.replace("_", "-"), type=type(campo.default), default=campo.default
        )
    entrenar(ConfigGRPO(**vars(ap.parse_args())))


if __name__ == "__main__":
    main()
