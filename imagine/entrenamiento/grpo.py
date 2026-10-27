"""GRPO (Group Relative Policy Optimization) sobre un modelo base, como en DeepSeek-R1-Zero.

Por cada fragmento se muestrean G imaginaciones, se puntúan contra el corpus y la
ventaja de cada una es su recompensa normalizada dentro del grupo (sin crítico):

    A_i = (r_i − media(r)) / std(r)

La pérdida es el objetivo recortado de PPO más una penalización KL (estimador k3)
contra el modelo base congelado. Se entrena con LoRA para que entre
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

from imagine.datos.archivos import TRAIN, agregar_jsonl, ruta_del_proyecto
from imagine.datos.repositorio import leer_ejemplos
from imagine.dominio.corpus import Ejemplo
from imagine.dominio.escritura import TEMPERATURA_MUESTREO, TOKENS_IMAGINACION
from imagine.dominio.recompensa import Ganancia, recompensa
from imagine.dominio.seleccion import mejor_imaginacion
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
    grupo: int = 8  # G: imaginaciones por fragmento
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


def perdida_grpo(model, cache, ultimo, comps, mascara, ventajas, base_lp, viejo_lp, eps, beta, g):
    lp = logprobs_completacion(model, cache, ultimo, comps)
    ratio = mx.exp(lp - viejo_lp)
    pg = -mx.minimum(ratio * ventajas, mx.clip(ratio, 1 - eps, 1 + eps) * ventajas)
    # KL (estimador k3) contra el modelo base: el ancla de la que no debe alejarse.
    kl = mx.exp(base_lp - lp) - (base_lp - lp) - 1
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


@dataclass
class Grupo:
    """Las G imaginaciones de un fragmento, juzgadas contra su referencia."""

    prompt_ids: list[int]
    completaciones: list[list[int]]
    recompensas: list[float]
    ganancias: list[Ganancia]
    mejor_imaginacion: str | None  # para el log; None si ninguna tiene formato válido


def muestrear_grupo(model, juez, tok, ejemplo: Ejemplo, conf: ConfigGRPO) -> Grupo:
    p_ids, generadas = escritor.imaginar(
        model, tok, ejemplo.fragmento, conf.grupo, conf.max_tokens, conf.temperatura
    )
    imaginaciones = [g.imaginacion for g in generadas]
    g = puntuacion.ganancias(juez, tok, p_ids, imaginaciones, ejemplo.referencia)
    r = [recompensa(x) for x in g]
    mejor = mejor_imaginacion(imaginaciones, g)
    return Grupo(p_ids, [gen.ids for gen in generadas], r, g, mejor.texto if mejor else None)


def gradientes(model, base, tok, grupo: Grupo, ventajas: mx.array, conf: ConfigGRPO):
    """Gradiente de la pérdida GRPO del grupo, por micro-lotes. Devuelve (grads, loss, kl)."""
    p_ids, comps = grupo.prompt_ids, grupo.completaciones
    grad_fn = nn.value_and_grad(model, perdida_grpo)
    grads, loss, kl, n_tokens = None, 0.0, 0.0, 0
    for i in range(0, len(comps), conf.micro_lote):
        sub = comps[i : i + conf.micro_lote]
        ids = rellenar(sub, tok.eos_token_id)
        mascara = mx.arange(ids.shape[1])[None] < mx.array([len(c) for c in sub])[:, None]
        mascara = mascara.astype(mx.float32)
        n = len(sub)
        base_lp = logprobs_completacion(base, prellenar(base, p_ids, n), p_ids[-1], ids)
        # Una sola actualización por lote (on-policy): π_old = π_θ sin gradiente.
        viejo_lp = logprobs_completacion(model, prellenar(model, p_ids, n), p_ids[-1], ids)
        mx.eval(base_lp, viejo_lp)

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
            base_lp,
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
    return grads, loss, kl / n_tokens


def paso_grupo(model, juez, base, tok, ejemplo: Ejemplo, conf: ConfigGRPO):
    """Muestrea y puntúa un grupo; si hay señal, calcula gradientes. → (grads o None, métricas)."""
    grupo = muestrear_grupo(model, juez, tok, ejemplo, conf)
    r, g, comps = grupo.recompensas, grupo.ganancias, grupo.completaciones
    metricas = {
        "recompensa": sum(r) / len(r),
        "formato": sum(x is not None for x in g) / len(g),
        "ganancia": [x for x in g if x is not None],
        "largo": sum(len(c) for c in comps) / len(comps),
        "mejor_imaginacion": grupo.mejor_imaginacion,
    }
    ventajas = ventajas_de_grupo(r)
    if ventajas is None:
        return None, metricas
    grads, loss, kl = gradientes(model, base, tok, grupo, ventajas, conf)
    metricas.update(loss=loss, kl=kl)
    return grads, metricas


def entrenar(conf: ConfigGRPO) -> None:
    # Generador propio: el estado global de random lo tocan librerías al importarse.
    rng = random.Random(conf.semilla)
    mx.random.seed(conf.semilla)
    salida = ruta_del_proyecto(conf.salida)
    salida.mkdir(parents=True, exist_ok=True)
    config_lora = adaptadores.config_lora(MODELO_BASE, conf.capas, conf.rango, conf.escala)

    ejemplos = leer_ejemplos(conf.datos)
    model, tok = cargar()
    base, _ = cargar()
    base.freeze()
    # Hoy el juez es el mismo modelo base congelado (ADR 0001).
    juez = base
    agregar_lora(model, conf.capas, conf.rango, conf.escala)
    print(f"parámetros LoRA: {millones_entrenables(model):.1f}M")
    opt = optim.AdamW(learning_rate=conf.lr, weight_decay=0.0)

    log = open(salida / "log.jsonl", "a", encoding="utf-8")
    for paso in range(1, conf.pasos + 1):
        t0 = time.time()
        acum, n_grads, ms = None, 0, []
        for ej in rng.sample(ejemplos, conf.prompts_por_paso):
            grads, m = paso_grupo(model, juez, base, tok, ej, conf)
            ms.append(m)
            if grads is None:
                continue
            acum = grads if acum is None else tree_map(mx.add, acum, grads)
            n_grads += 1
        if acum is not None:
            opt.update(model, tree_map(lambda g: g / n_grads, acum))
            mx.eval(model.trainable_parameters(), opt.state)

        ganancias = [x for m in ms for x in m["ganancia"]]
        fila = {
            "paso": paso,
            "recompensa": sum(m["recompensa"] for m in ms) / len(ms),
            "formato": sum(m["formato"] for m in ms) / len(ms),
            "ganancia": sum(ganancias) / len(ganancias) if ganancias else None,
            "largo": sum(m["largo"] for m in ms) / len(ms),
            "kl": sum(m.get("kl", 0) for m in ms) / len(ms),
            "grupos_con_señal": n_grads,
            "seg": round(time.time() - t0, 1),
            "mem_gb": round(mx.get_peak_memory() / 1e9, 1),
            "mejor_imaginacion": ms[0]["mejor_imaginacion"],
        }
        agregar_jsonl(log, fila)
        d = f"{fila['ganancia']:+.4f}" if fila["ganancia"] is not None else "  —   "
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
