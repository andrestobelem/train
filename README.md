# imagine

Adaptación de la receta de [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (RL puro con GRPO y recompensas basadas en reglas sobre un modelo base) a la escritura literaria en lugar de la matemática.

| DeepSeek-R1-Zero | imagine |
|---|---|
| DeepSeek-V3-Base (671B) | [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) (feb. 2026, sin instrucciones ni razonamiento) |
| `<think> … </think>` antes de responder | `<imagine> … </imagine>` antes de continuar la novela |
| Recompensa de exactitud: ¿la respuesta es correcta? | Recompensa de corpus: cuánto mejora lo imaginado la predicción de lo que el autor escribió de verdad |
| Recompensa de formato | Recompensa de formato: cierra `</imagine>`, no está vacío, no reabre la etiqueta |
| GRPO | GRPO (objetivo recortado + KL k3 contra el modelo de referencia), con LoRA sobre MLX |

## Recompensa de corpus

Cada ejemplo es un fragmento de una novela en español (Project Gutenberg) y el texto que le sigue, que el modelo nunca ve. Con el modelo de referencia congelado:

```
Δ = log p(referencia | fragmento, imaginación) − log p(referencia | fragmento, imaginación vacía)   (media por token)
r = 1 + tanh(4·Δ)   si el formato es válido, 0 si no
```

La imaginación no ve la referencia, así que sólo suma si anticipa lo que de verdad pasa. Se verifica contra el corpus sin entrenar un modelo de recompensa, por la misma razón por la que R1 evita las recompensas neuronales.

- Entrenamiento: *Don Quijote* (Cervantes) y *Fortunata y Jacinta* (Pérez Galdós).
- Test: *Niebla* (Unamuno), un libro que no aparece en el entrenamiento.

Una diferencia con R1-Zero: el prompt incluye un ejemplo (one-shot). Sin él, el modelo de 0.8B cierra la etiqueta sólo ~5 % de las veces; con él, ~90 % a temperatura 0.8.

## Resultados

300 pasos de GRPO (4 fragmentos × G=8 por paso, lr 2e-6, LoRA rango 16), ~3 h en una M2 Max, con un checkpoint cada 50 pasos. Cada modelo se evaluó sobre 40 fragmentos de *Niebla* con 4 muestras por fragmento, dos veces: con `imagine.evaluar` y con Inspect (`evals/niebla_flow.py`). La columna Δ está en nats/token.

| Modelo | Formato (evaluar / Inspect) | Δ evaluar | Δ Inspect | Imaginaciones que ayudan (evaluar / Inspect) |
|---|---|---|---|---|
| base | 86 % / 84 % | −0,021 | −0,024 | 30 % / 28 % |
| paso 50 | 98 % / 98 % | +0,008 | +0,010 | 56 % / 57 % |
| paso 100 | 98 % / 96 % | +0,013 | +0,016 | 61 % / 69 % |
| paso 150 | 99 % / 94 % | +0,021 | +0,026 | 70 % / 75 % |
| paso 200 | 100 % / 99 % | +0,025 | +0,018 | 76 % / 67 % |
| paso 250 | 99 % / 100 % | +0,023 | +0,027 | 74 % / 71 % |
| paso 300 | 98 % / 98 % | +0,021 | +0,021 | 70 % / 67 % |

- Antes del entrenamiento, imaginar empeoraba la predicción del texto real. Después la mejora, en un libro que el modelo no vio: de −0,02 a +0,02 nats/token.
- La mejora crece hasta el paso ~150 y después se estanca. Entre los pasos 150 y 300 las diferencias (~0,005) están dentro del ruido: el error estándar de Δ es ±0,003 por muestra, y es mayor porque las 4 muestras de un fragmento están correlacionadas.
- Las dos evaluaciones usan muestras distintas y coinciden dentro de ese ruido.
- La prosa sigue siendo la de un modelo de 0.8B.

## Uso

```bash
uv sync
uv run python -m imagine.corpus                            # descarga y arma data/
uv run python -m imagine.entrenar --salida runs/imagine300 # entrenamiento GRPO
uv run python -m imagine.evaluar --adaptadores runs/imagine300/paso_00150
uv run python -m imagine.generar --adaptadores runs/imagine300/paso_00150 "Fragmento…"
```

### Con Inspect AI e Inspect Flow

El proyecto registra un proveedor `mlx/` en Inspect AI. El nombre del modelo es un repo de Hugging Face o una carpeta de checkpoint:

```bash
uv run inspect eval imagine/inspect_tarea.py@niebla --model mlx/Qwen/Qwen3.5-0.8B-Base
uv run inspect eval imagine/inspect_tarea.py@niebla --model mlx/runs/imagine300/paso_00150
uv run flow run evals/niebla_flow.py      # base + todos los checkpoints de IMAGINE_RUN
uv run inspect view --log-dir logs/niebla
```

Pensado para Apple Silicon (MLX). Las métricas de entrenamiento quedan en `runs/<nombre>/log.jsonl`.
