"""Tablas de resultados a partir de los logs de Inspect, una por tarea.

    uv run python evals/resumen.py logs/niebla
    uv run python evals/resumen.py logs/prosa
"""

import sys
from collections import defaultdict

from inspect_ai.log import list_eval_logs, read_eval_log


def filas_por_tarea(carpeta: str) -> dict[str, list[tuple[str, dict]]]:
    tareas = defaultdict(list)
    for info in list_eval_logs(carpeta):
        log = read_eval_log(info, header_only=True)
        if log.status != "success" or not log.results:
            continue
        # Una sola métrica por clave (media_valida): se toma su valor.
        valores = {s.name: next(iter(s.metrics.values())).value for s in log.results.scores}
        nombre = log.eval.model.removeprefix("mlx/")
        if log.eval.task_args.get("imaginar") is False:
            nombre += " (sin imaginar)"
        tareas[log.eval.task].append((nombre, valores))
    return tareas


def imprimir(tarea: str, filas: list[tuple[str, dict]]) -> None:
    claves = list(filas[0][1])
    ancho = max(len(n) for n, _ in filas)
    print(f"\n{tarea}")
    print(f"{'modelo':{ancho}s}  " + "  ".join(f"{k:>10s}" for k in claves))
    for nombre, valores in sorted(filas):
        celdas = (f"{valores.get(k, float('nan')):>10.4f}" for k in claves)
        print(f"{nombre:{ancho}s}  " + "  ".join(celdas))


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "logs/niebla"
    tareas = filas_por_tarea(carpeta)
    if not tareas:
        print(f"sin logs completos en {carpeta}")
    for tarea, filas in sorted(tareas.items()):
        imprimir(tarea, filas)


if __name__ == "__main__":
    main()
