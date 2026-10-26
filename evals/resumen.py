"""Tabla de resultados a partir de los logs de Inspect.

    uv run python evals/resumen.py logs/niebla
    uv run python evals/resumen.py logs/prosa
"""

import sys

from inspect_ai.log import list_eval_logs, read_eval_log


def main():
    carpeta = sys.argv[1] if len(sys.argv) > 1 else "logs/niebla"
    filas = []
    for info in list_eval_logs(carpeta):
        log = read_eval_log(info, header_only=True)
        if log.status != "success" or not log.results:
            continue
        # Una sola métrica por clave (media_valida): se toma su valor.
        valores = {s.name: next(iter(s.metrics.values())).value for s in log.results.scores}
        nombre = log.eval.model.removeprefix("mlx/")
        if log.eval.task_args.get("imaginar") is False:
            nombre += " (sin imaginar)"
        filas.append((nombre, valores))
    if not filas:
        print(f"sin logs completos en {carpeta}")
        return
    claves = list(filas[0][1])
    ancho = max(len(n) for n, _ in filas)
    print(f"{'modelo':{ancho}s}  " + "  ".join(f"{k:>10s}" for k in claves))
    for nombre, valores in sorted(filas):
        print(f"{nombre:{ancho}s}  " + "  ".join(f"{valores[k]:>10.4f}" for k in claves))


if __name__ == "__main__":
    main()
