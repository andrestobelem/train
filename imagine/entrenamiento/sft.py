"""Selección + ajuste supervisado, la etapa que sigue al RL en DeepSeek-R1.

1. muestrear: por cada fragmento de entrenamiento, el escritor de RL imagina G veces y
   la selección se queda con la imaginación de mayor ganancia, si ayuda (ver GLOSSARY.md).
2. entrenar: ajuste supervisado sobre fragmento → imaginación elegida → referencia. Así
   el escritor aprende a continuar como el autor después de imaginar, algo que el RL
   nunca le enseñó: sólo premiaba la imaginación.

    uv run python -m imagine.entrenamiento.sft muestrear --adaptadores runs/imagine300/paso_00250
    uv run python -m imagine.entrenamiento.sft entrenar --adaptadores runs/imagine300/paso_00250
"""

import argparse
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

from imagine.datos.archivos import SFT, TRAIN, agregar_jsonl, ruta_del_proyecto
from imagine.datos.repositorio import agregar_supervisado, leer_ejemplos, leer_supervisados
from imagine.dominio.corpus import EjemploSupervisado
from imagine.dominio.escritura import TEMPERATURA_MUESTREO, TOKENS_IMAGINACION, prompt
from imagine.dominio.seleccion import seleccionar
from imagine.ejecucion import escritor, puntuacion
from imagine.ejecucion.lm import logprobs_completacion, prellenar
from imagine.ejecucion.modelos import cargar
from imagine.persistencia import adaptadores, catalogo

from .lora import entrenar_solo_lora, millones_entrenables


def muestrear(
    origen: str,
    datos: str,
    salida_datos: str,
    n: int,
    grupo: int,
    max_tokens: int,
    temperatura: float,
    semilla: int,
) -> None:
    """La selección: guarda la imaginación elegida de cada fragmento. Se puede retomar."""
    # Generador propio: el estado global de random lo tocan librerías al importarse.
    rng = random.Random(semilla)
    mx.random.seed(semilla)
    ejemplos = rng.sample(leer_ejemplos(datos), n)
    salida = ruta_del_proyecto(salida_datos)
    hechos = len(leer_supervisados(salida)) if salida.exists() else 0
    progreso = salida.with_name(salida.name + ".progreso")
    inicio = int(progreso.read_text()) if progreso.exists() else 0
    print(f"retomando desde el fragmento {inicio} ({hechos} ejemplos guardados)")

    model, tok = cargar(origen)
    juez, _ = cargar()
    with open(salida, "a", encoding="utf-8") as f:
        for i in range(inicio, len(ejemplos)):
            ejemplo = ejemplos[i]
            t0 = time.time()
            p_ids, generadas = escritor.imaginar(
                model, tok, ejemplo.fragmento, grupo, max_tokens, temperatura
            )
            imaginaciones = [g.imaginacion for g in generadas]
            ganancias = puntuacion.ganancias(juez, tok, p_ids, imaginaciones, ejemplo.referencia)
            elegida = seleccionar(imaginaciones, ganancias)
            if elegida is not None:
                agregar_supervisado(f, EjemploSupervisado(ejemplo, *elegida))
                hechos += 1
            progreso.write_text(str(i + 1))
            texto = f"{elegida[1]:+.4f}" if elegida else "  —   "
            print(
                f"{i + 1}/{len(ejemplos)} | mejor Δ {texto} | guardados {hechos} | "
                f"{time.time() - t0:.1f}s"
            )


def perdida_sft(model, cache, ultimo, comps):
    return -logprobs_completacion(model, cache, ultimo, comps).mean()


def _tokens(tok, supervisado: EjemploSupervisado) -> tuple[list[int], list[int]]:
    """(prompt, objetivo) de un ejemplo supervisado."""
    p_ids = tok.encode(prompt(supervisado.ejemplo.fragmento))
    return p_ids, tok.encode(supervisado.objetivo, add_special_tokens=False)


def _perdida_media(model, tok, filas: list[EjemploSupervisado]) -> float:
    total = 0.0
    for fila in filas:
        p_ids, c_ids = _tokens(tok, fila)
        cache = prellenar(model, p_ids, 1)
        total += perdida_sft(model, cache, p_ids[-1], mx.array([c_ids])).item()
    return total / len(filas)


def entrenar(
    origen: str, salida_datos: str, salida: str, epocas: int, lr: float, acumular: int, semilla: int
) -> None:
    """SFT de los adaptadores de origen; guarda uno por época en salida/epoca_N."""
    # Generador propio: el estado global de random lo tocan librerías al importarse.
    rng = random.Random(semilla)
    filas = leer_supervisados(salida_datos)
    rng.shuffle(filas)
    n_val = max(1, len(filas) // 20)
    val, train = filas[:n_val], filas[n_val:]
    print(f"train {len(train)} | val {len(val)}")

    model, tok = cargar(origen)
    entrenar_solo_lora(model)
    print(f"parámetros entrenables: {millones_entrenables(model):.1f}M")
    # Misma configuración de LoRA que los adaptadores de partida.
    config_lora = adaptadores.leer_config(ruta_del_proyecto(origen))
    opt = optim.AdamW(learning_rate=lr, weight_decay=0.0)
    grad_fn = nn.value_and_grad(model, perdida_sft)

    destino = ruta_del_proyecto(salida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    log = open(destino.parent / f"{destino.name}_log.jsonl", "a", encoding="utf-8")
    print(f"época 0 | val {_perdida_media(model, tok, val):.4f}")
    for epoca in range(1, epocas + 1):
        rng.shuffle(train)
        acum, perdidas, t0 = None, [], time.time()
        for k, fila in enumerate(train, 1):
            p_ids, c_ids = _tokens(tok, fila)
            cache = prellenar(model, p_ids, 1)
            model.train()
            loss, g = grad_fn(model, cache, p_ids[-1], mx.array([c_ids]))
            model.eval()
            acum = g if acum is None else tree_map(mx.add, acum, g)
            mx.eval(acum, loss)
            perdidas.append(loss.item())
            if k % acumular == 0 or k == len(train):
                opt.update(model, tree_map(lambda x: x / acumular, acum))
                mx.eval(model.trainable_parameters(), opt.state)
                acum = None
            if k % 50 == 0:
                media = sum(perdidas[-50:]) / len(perdidas[-50:])
                print(
                    f"época {epoca} | {k}/{len(train)} | loss {media:.4f} | {time.time() - t0:.0f}s"
                )
        fila_log = {
            "epoca": epoca,
            "loss": sum(perdidas) / len(perdidas),
            "val": _perdida_media(model, tok, val),
        }
        agregar_jsonl(log, fila_log)
        print(f"época {epoca} | train {fila_log['loss']:.4f} | val {fila_log['val']:.4f}")
        adaptadores.guardar(model, destino / f"epoca_{epoca}", config_lora)
    adaptadores.guardar(model, destino, config_lora)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nombre in ["muestrear", "entrenar"]:
        p = sub.add_parser(nombre)
        p.add_argument(
            "--adaptadores",
            default=catalogo.RL,
            help="modelo de RL del que se parte",
        )
        p.add_argument("--salida-datos", default=str(SFT))
        p.add_argument("--semilla", type=int, default=1)
    m = sub.choices["muestrear"]
    m.add_argument("--datos", default=str(TRAIN))
    m.add_argument("-n", type=int, default=600, help="fragmentos a muestrear")
    m.add_argument("--grupo", type=int, default=8)
    m.add_argument("--max-tokens", type=int, default=TOKENS_IMAGINACION)
    m.add_argument("--temperatura", type=float, default=TEMPERATURA_MUESTREO)
    e = sub.choices["entrenar"]
    e.add_argument("--salida", default=catalogo.SALIDA_SFT)
    e.add_argument("--epocas", type=int, default=2)
    e.add_argument("--lr", type=float, default=5e-6)
    e.add_argument("--acumular", type=int, default=4, help="ejemplos por actualización")
    a = ap.parse_args()
    if a.cmd == "muestrear":
        muestrear(
            a.adaptadores,
            a.datos,
            a.salida_datos,
            a.n,
            a.grupo,
            a.max_tokens,
            a.temperatura,
            a.semilla,
        )
    else:
        entrenar(a.adaptadores, a.salida_datos, a.salida, a.epocas, a.lr, a.acumular, a.semilla)


if __name__ == "__main__":
    main()
