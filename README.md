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

300 pasos de GRPO (4 fragmentos × G=8 por paso, lr 2e-6, LoRA rango 16), ~3 h en una M2 Max, con un checkpoint cada 50 pasos. Cada modelo se evaluó sobre 40 fragmentos de *Niebla* con 4 muestras por fragmento, dos veces: con el script `evaluar` original (hoy reemplazado por la tarea `imagine/niebla`) y con Inspect (`evals/niebla_flow.py`). La columna Δ está en nats/token.

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

## Segunda etapa: rejection sampling + SFT

Como en DeepSeek-R1, después del RL viene una etapa supervisada (`imagine.sft`):

1. El modelo de RL (paso 250) imagina 8 veces sobre 600 fragmentos de entrenamiento. Por cada fragmento se queda la imaginación con mayor Δ positivo, verificada contra el corpus. Quedan 569 ejemplos.
2. Ajuste supervisado sobre fragmento → mejor imaginación → **texto real del autor**. El RL sólo premiaba la imaginación; acá el modelo aprende también a escribir la continuación como Cervantes o Galdós.

La val loss baja de 3,33 a 3,03 en la época 1 y sube en la 2, así que se usa `runs/sft/epoca_1`.

### Evaluación final en *Niebla* (Inspect)

Imaginación (tarea `imagine/niebla`, 40 fragmentos × 4 muestras; Δ en nats/token):

| Modelo | Formato | Δ | Imaginaciones que ayudan |
|---|---|---|---|
| base | 84 % | −0,026 | 28 % |
| RL paso 150 | 94 % | +0,026 | 75 % |
| RL paso 300 | 97 % | +0,021 | 67 % |
| RL + SFT | 99 % | +0,030 | 77 % |

Prosa de la continuación (tarea `imagine/prosa`, 40 fragmentos × 2 muestras):

| Modelo | Repetición ↓ | Diversidad ↑ | Fluidez ↑ | chrF con el autor ↑ | Palabras |
|---|---|---|---|---|---|
| base, sin imaginar | 0,016 | 0,968 | −2,20 | 0,156 | 39 |
| base, imaginando | 0,021 | 0,951 | −2,08 | 0,135 | 32 |
| RL paso 150 | 0,006 | 0,969 | −3,55 | 0,203 | 54 |
| RL paso 300 | 0,002 | 0,982 | −3,62 | 0,215 | 64 |
| RL + SFT | 0,011 | 0,967 | −2,87 | 0,169 | 38 |

- El SFT mantiene (y mejora un poco) la imaginación del RL.
- El SFT recupera parte de la fluidez que perdió el RL y escribe continuaciones del largo de las del autor. El chrF alto del RL viene en parte de que escribe más.
- Con un largo similar al del modelo base sin imaginar, el SFT se parece más al autor (chrF 0,169 contra 0,156).
- La fluidez se mide sin la imaginación en el contexto, así que penaliza continuaciones que siguen lo imaginado. La prosa sigue siendo la de un modelo de 0.8B.

## Lenguaje del dominio

El glosario del proyecto está en [GLOSSARY.md](GLOSSARY.md): fragmento, imaginación, continuación, referencia, ganancia, recompensa, selección, etc. El código usa esos mismos términos. Las decisiones difíciles de revertir están en [docs/adr/](docs/adr/):

- [0001](docs/adr/0001-recompensa-por-ganancia-con-modelo-base.md): la recompensa es la ganancia medida por el modelo base congelado.
- [0002](docs/adr/0002-plantilla-con-un-ejemplo.md): la plantilla incluye un ejemplo, a diferencia de R1-Zero.
- [0003](docs/adr/0003-gradiente-solo-por-lo-generado.md): el gradiente pasa sólo por lo que genera el escritor.
- [0004](docs/adr/0004-mlx-en-apple-silicon.md): se entrena y se sirve con MLX en Apple Silicon.

Lo que en logs y tablas viejas figura como `delta` es la **ganancia** (Δ). Desde el modelado del dominio, los logs nuevos de GRPO y las métricas de `imagine/niebla` la llaman `ganancia`. El formato de los jsonl de `data/` no cambió.

## Arquitectura

El código sigue los principios de [CONSTITUTION.md](CONSTITUTION.md). Cada paquete de `imagine/` tiene una sola responsabilidad, y las dependencias van de arriba hacia abajo:

| Paquete | Responsabilidad | Depende de |
|---|---|---|
| `dominio` | Modelo del dominio (Python puro): imaginación, ejemplo y ejemplo supervisado, recompensa, selección, métricas de prosa | — |
| `datos` | Corpus de Gutenberg; repositorio de ejemplos en jsonl | `dominio` |
| `persistencia` | Guardar y leer adaptadores LoRA; catálogo de los modelos entrenados | — |
| `ejecucion` | MLX: carga con caché única, hilo dedicado, muestreo, escritor, puntuación contra el corpus | `dominio`, `datos`, `persistencia` |
| `entrenamiento` | GRPO, rejection sampling + SFT, LoRA | `ejecucion` y anteriores |
| `integraciones.inspect` | Proveedor `mlx/` y tareas `imagine/niebla` e `imagine/prosa` | `ejecucion` y anteriores |
| `interfaz` | Línea de comandos y app de Gradio | `ejecucion` y anteriores |

Donde el código pide un modelo, se acepta un repo de Hugging Face (`Qwen/Qwen3.5-0.8B-Base`) o una carpeta de adaptadores del proyecto (`runs/sft/epoca_1`). Los modelos entrenados de referencia (el RL y el final) están en `imagine/persistencia/catalogo.py`. Los presupuestos de generación (tokens y temperaturas) están en `imagine/dominio/escritura.py`.

`tests/test_arquitectura.py` verifica que cada capa importe sólo las de abajo y que el dominio no dependa de MLX, Inspect ni Gradio:

```bash
uv run pytest
uv run ruff check imagine evals tests
```

## Uso

```bash
uv sync
uv run python -m imagine.datos.corpus                                    # descarga y arma data/
uv run python -m imagine.entrenamiento.grpo --salida runs/imagine300     # RL con GRPO
uv run python -m imagine.entrenamiento.sft muestrear --adaptadores runs/imagine300/paso_00250
uv run python -m imagine.entrenamiento.sft entrenar --adaptadores runs/imagine300/paso_00250
uv run python -m imagine.interfaz.generar "Fragmento…"                     # modelo final por defecto
uv run python -m imagine.interfaz.app                                    # Gradio en http://127.0.0.1:7860
```

### Con Inspect AI e Inspect Flow

El proyecto registra en Inspect AI el proveedor `mlx/` y las tareas `imagine/niebla` (imaginación) e `imagine/prosa` (calidad de la continuación):

```bash
uv run inspect eval imagine/niebla --model mlx/Qwen/Qwen3.5-0.8B-Base
uv run inspect eval imagine/niebla --model mlx/runs/sft/epoca_1
uv run inspect eval imagine/prosa --model mlx/Qwen/Qwen3.5-0.8B-Base -T imaginar=false
uv run flow run evals/niebla_flow.py      # base + todos los checkpoints de IMAGINE_RUN
uv run flow run evals/prosa_flow.py       # prosa: base, base imaginando y entrenados
uv run python evals/resumen.py logs/prosa # tablas desde los logs
uv run inspect view --log-dir logs/niebla
```

Pensado para Apple Silicon (MLX). Las métricas de entrenamiento quedan en `runs/<nombre>/log.jsonl`.
