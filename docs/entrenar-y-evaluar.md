# Entrenar y evaluar

Cómo reproducir el modelo final desde cero y evaluarlo. Los tiempos son los de una M2 Max con 64 GB. Para sólo usar un modelo ya entrenado, ver [correr-el-modelo.md](correr-el-modelo.md). Todo lo que producen estos pasos ya viene con el repo por Git LFS (ver [Modelos, datos y resultados](../README.md#modelos-datos-y-resultados)): sólo hace falta repetirlos para cambiar algo.

## 1. Corpus

```bash
uv run python -m imagine.datos.corpus
```

Descarga de Project Gutenberg los tres libros del corpus y arma los ejemplos (fragmento + referencia):

- `data/train.jsonl`: *Don Quijote* y *Fortunata y Jacinta*.
- `data/test.jsonl`: *Niebla*, el libro de prueba. Su autor no escribió ningún libro de entrenamiento.

El resultado es determinista: correrlo de nuevo produce los mismos archivos.

## 2. RL con GRPO (~3 h)

```bash
uv run python -m imagine.entrenamiento.grpo --salida runs/imagine300 --pasos 300
```

En cada paso, el escritor imagina 8 veces sobre cada uno de 4 fragmentos. El juez (el modelo base congelado) mide la ganancia de cada imaginación, y la recompensa decide qué se refuerza (ver [ADR 0001](adr/0001-recompensa-por-ganancia-con-modelo-base.md)). Se entrena sólo LoRA (rango 16) y se guarda un checkpoint cada 50 pasos en `paso_NNNNN/`. El pico de memoria es de ~12,4 GB.

Las opciones salen de `ConfigGRPO` en [`imagine/entrenamiento/grpo.py`](../imagine/entrenamiento/grpo.py) y se pasan con guiones, por ejemplo `--micro-lote 1` si falta memoria o `--lr 1e-6`. `--help` las lista todas.

Las métricas de cada paso quedan en `runs/imagine300/log.jsonl`: recompensa, ganancia, formato válido, KL y la mejor imaginación del grupo.

## 3. Selección y ajuste supervisado (~1 h)

```bash
uv run python -m imagine.entrenamiento.sft muestrear --adaptadores runs/imagine300/paso_00250
uv run python -m imagine.entrenamiento.sft entrenar --adaptadores runs/imagine300/paso_00250
```

1. **muestrear** (~45 min): el modelo de RL imagina 8 veces sobre 600 fragmentos de entrenamiento y la selección se queda, por cada uno, con la imaginación de mayor ganancia, si ayuda. Quedan ~570 ejemplos supervisados en `data/sft.jsonl`. Si se corta, vuelve a correrlo y retoma desde donde quedó.
2. **entrenar** (~10 min por época): ajuste supervisado sobre fragmento → imaginación elegida → referencia. Guarda cada época en `runs/sft/epoca_N` y el log en `runs/sft_log.jsonl`.

El modelo final es la época con menor pérdida de validación. En la corrida publicada fue la 1, así que `catalogo.FINAL` apunta a `runs/sft/epoca_1`. Si entrenás de nuevo, revisá el log y actualizá [`imagine/persistencia/catalogo.py`](../imagine/persistencia/catalogo.py).

## 4. Evaluar con Inspect AI

El proyecto registra en Inspect AI el proveedor `mlx/` y dos tareas sobre *Niebla*:

- `imagine/niebla`: la imaginación. Mide el formato válido, la ganancia y las imaginaciones que ayudan.
- `imagine/prosa`: la continuación. Mide repetición, diversidad, letras ajenas, fluidez, parecido con el autor y largo.

```bash
uv run inspect eval imagine/niebla --model mlx/runs/sft/epoca_1
uv run inspect eval imagine/niebla --model mlx/Qwen/Qwen3.5-0.8B-Base
uv run inspect eval imagine/prosa --model mlx/Qwen/Qwen3.5-0.8B-Base -T imaginar=false
uv run inspect view --log-dir logs
```

Las tareas reciben parámetros con `-T`: `n` (fragmentos, por defecto 40), `intentos`, `temperatura` y, en `prosa`, `imaginar`.

### Varias evaluaciones con Inspect Flow

```bash
uv run flow run evals/niebla_flow.py      # base + todos los checkpoints de IMAGINE_RUN
uv run flow run evals/prosa_flow.py       # base sin y con imaginación, RL y final
uv run python evals/resumen.py logs/niebla
uv run python evals/resumen.py logs/prosa
```

- `IMAGINE_RUN` elige qué corrida de GRPO evalúa `niebla_flow.py`. Por defecto es `runs/imagine300`.
- `IMAGINE_MODELOS` elige qué modelos entrenados compara `prosa_flow.py`, separados por comas.
- `flow run --dry-run` muestra qué se correría sin correrlo.

## Antes de commitear

```bash
uv run ruff format imagine evals tests
uv run ruff check imagine evals tests
uv run pytest
```

`tests/test_arquitectura.py` verifica que cada capa de `imagine/` importe sólo las que le corresponden (ver la tabla del [README](../README.md#arquitectura)).
