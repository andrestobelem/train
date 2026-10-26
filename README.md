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

## Uso

```bash
uv sync
uv run python -m imagine.corpus                         # descarga y arma data/
uv run python -m imagine.entrenar --salida runs/imagine # entrenamiento GRPO
uv run python -m imagine.evaluar                        # modelo base sobre Niebla
uv run python -m imagine.evaluar --adaptadores runs/imagine
uv run python -m imagine.generar --adaptadores runs/imagine "Fragmento…"
```

Pensado para Apple Silicon (MLX). Las métricas de entrenamiento quedan en `runs/<nombre>/log.jsonl`.
