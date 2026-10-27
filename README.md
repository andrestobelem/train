# imagine

> **[Ver ejemplos de lo que escribe →](docs/ejemplos.md)** Una selección para leer sin instalar nada y sin saber de tecnología.

## En pocas palabras

Enseñamos a una inteligencia artificial chica a **imaginar antes de escribir**. Le damos un pedazo de una novela en español y, antes de seguirla, el modelo anota en pocas líneas qué sienten los personajes y qué va a pasar. Recién después escribe el párrafo siguiente.

**Con qué aprendió.** Con dos novelas clásicas en español, de dominio público y bajadas de Project Gutenberg: [*Don Quijote*](https://www.gutenberg.org/ebooks/2000), de Miguel de Cervantes, y [*Fortunata y Jacinta*](https://www.gutenberg.org/ebooks/17013), de Benito Pérez Galdós. Para probarlo usamos una tercera que nunca vio, de un autor que tampoco estaba en el entrenamiento: [*Niebla*](https://www.gutenberg.org/ebooks/49836), de Miguel de Unamuno.

¿Cómo aprende a imaginar bien? Comparamos lo que imaginó con lo que el autor escribió de verdad, que el modelo nunca vio. Si lo imaginado ayuda a anticipar lo que pasa en la novela, el modelo recibe un premio, y así aprende a imaginar mejor. Es la misma idea con la que DeepSeek-R1 aprendió a «pensar» antes de resolver problemas de matemática, aplicada a la literatura.

**El resultado.** Antes de entrenarlo, lo que imaginaba lo despistaba más de lo que lo ayudaba. Después, en *Niebla*, que nunca había leído, imagina cosas que sí ayudan a anticipar lo que escribió el autor en tres de cada cuatro casos. Es un modelo muy chico, que corre en una notebook, así que todavía escribe como un aprendiz: arma bien las frases, pero pierde el hilo rápido.

**Para probarlo.** Se puede usar en el navegador o escribir una historia por turnos con él: vos escribís un párrafo, el modelo sigue, y así. Las instrucciones están en [Empezar en una computadora nueva](#empezar-en-una-computadora-nueva).

**Para ver resultados sin instalar nada.** Los [ejemplos](docs/ejemplos.md) muestran lo mejor que escribió. El [apéndice](docs/apendice-pruebas.md) explica paso a paso, sin tecnicismos, cómo se pone a prueba el modelo. Muestra cuatro fragmentos de *Niebla* con lo que imaginó y escribió el modelo antes y después de entrenarlo, al lado de lo que escribió Unamuno.

## Empezar en una computadora nueva

Hace falta una **Mac con chip Apple** (M1 o posterior): el modelo corre con [MLX](https://github.com/ml-explore/mlx), que no anda en Windows, Linux ni Macs con Intel ([ADR 0004](docs/adr/0004-mlx-en-apple-silicon.md)). Los comandos van en la Terminal.

**1. Instalar las herramientas.** Con [Homebrew](https://brew.sh/):

```bash
brew install uv git-lfs
git lfs install
```

- [uv](https://docs.astral.sh/uv/) instala Python y las dependencias del proyecto.
- [Git LFS](https://git-lfs.com/) baja los archivos grandes (los modelos y los datos). `git lfs install` se corre una sola vez por computadora, y **antes de clonar**: si no, los modelos quedan como punteros de texto (ver [Si algo falla](#si-algo-falla)).

**2. Clonar el repo.** Baja el código y, con Git LFS, también los modelos, los datos y los resultados (~470 MB):

```bash
git clone https://github.com/andrestobelem/train.git
cd train
```

**3. Instalar las dependencias:**

```bash
uv sync
```

**4. Verificar que todo anda.** Los tests no cargan el modelo, así que tardan unos segundos:

```bash
uv run pytest
```

**5. Usarlo.** La primera vez se baja de Hugging Face el modelo base (1,7 GB); después funciona sin conexión.

```bash
uv run imagine app             # abre la app en el navegador
uv run imagine cuatro-manos    # una historia por turnos en la terminal
uv run imagine generar "Augusto salió a la calle sin saber adónde ir."
```

Qué hace cada comando y sus opciones: [docs/correr-el-modelo.md](docs/correr-el-modelo.md).

### Qué trae el repo

Además del código y la documentación, el repo trae por Git LFS todo lo que produjo el proyecto. Así se puede seguir trabajando sin repetir nada:

| Carpeta | Qué hay | Tamaño |
|---|---|---|
| `runs/sft/epoca_1/` | El **modelo final**: el que usan la app y la terminal | 41 MB |
| `runs/` | Los demás modelos entrenados: checkpoints de RL (`imagine300/`) y del ajuste supervisado (`sft/`), y los registros de cada entrenamiento | 620 MB en total |
| `data/` | El corpus (los libros de Gutenberg y los ejemplos de entrenamiento y prueba) y los ejemplos supervisados de la selección | 19 MB |
| `logs/` | Los logs de las evaluaciones de Inspect; se ven con `uv run inspect view --log-dir logs` | 8 MB |

Lo único que no trae es el modelo base de Qwen, porque es de otros y pesa 1,7 GB: se baja solo la primera vez y queda en `~/.cache/huggingface`.

Todo lo de esas carpetas se puede regenerar desde cero siguiendo [docs/entrenar-y-evaluar.md](docs/entrenar-y-evaluar.md), pero lleva horas.

### Si algo falla

| Qué pasa | Qué hacer |
|---|---|
| `imagine` dice «Los modelos vienen con el repo por Git LFS, pero no se bajaron» | Se clonó sin Git LFS. En la carpeta del repo: `git lfs install` y `git lfs pull` |
| `brew: command not found` | Falta Homebrew: se instala desde [brew.sh](https://brew.sh/) |
| Querés bajar sólo el modelo final, no todo | `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/andrestobelem/train.git` y, dentro, `git lfs pull --include "runs/sft/epoca_1/*"` |
| Falla una descarga con `CERTIFICATE_VERIFY_FAILED` | Un proxy de la red intercepta la conexión: probá desde otra red |
| Otro error al usar el modelo | La tabla de [problemas comunes](docs/correr-el-modelo.md#problemas-comunes) |

El modelo final también está en un repo privado de Hugging Face, `andrestobelem/imagine-qwen3.5-0.8b`, como alternativa a Git LFS.

### Seguir trabajando

- **Antes de commitear:** `uv run ruff format imagine evals tests`, `uv run ruff check imagine evals tests` y `uv run pytest`.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) con scope y en español, uno por cambio lógico.
- **Con un agente como Claude Code:** [AGENTS.md](AGENTS.md) le dice qué leer antes: los principios de diseño, el glosario y las decisiones de diseño. La conversación con el agente no está en el repo, pero lo que se decidió en ella sí: en el glosario, los ADR y esta documentación.

## Uso

| Para… | Comando | Guía |
|---|---|---|
| Probar el modelo en el navegador | `uv run imagine app` | [correr-el-modelo.md](docs/correr-el-modelo.md#la-app) |
| Escribir a cuatro manos en la terminal | `uv run imagine cuatro-manos` | [correr-el-modelo.md](docs/correr-el-modelo.md#a-cuatro-manos) |
| Continuar un fragmento en la terminal | `uv run imagine generar "…"` | [correr-el-modelo.md](docs/correr-el-modelo.md#un-fragmento) |
| Elegir cómo se le presenta la escena (12 plantillas) | `uv run imagine plantillas` | [correr-el-modelo.md](docs/correr-el-modelo.md#plantillas) |
| Armar el corpus | `uv run python -m imagine.datos.corpus` | [entrenar-y-evaluar.md](docs/entrenar-y-evaluar.md#1-corpus) |
| Entrenar con GRPO | `uv run python -m imagine.entrenamiento.grpo --salida runs/imagine300` | [entrenar-y-evaluar.md](docs/entrenar-y-evaluar.md#2-rl-con-grpo-3-h) |
| Selección + SFT | `uv run python -m imagine.entrenamiento.sft muestrear` / `entrenar` | [entrenar-y-evaluar.md](docs/entrenar-y-evaluar.md#3-selección-y-ajuste-supervisado-1-h) |
| Evaluar en *Niebla* | `uv run inspect eval imagine/niebla --model mlx/runs/sft/epoca_1` | [entrenar-y-evaluar.md](docs/entrenar-y-evaluar.md#4-evaluar-con-inspect-ai) |

## Documentación

| Documento | Para qué |
|---|---|
| [docs/ejemplos.md](docs/ejemplos.md) | Para no técnicos: una selección de lo mejor que escribió el modelo |
| [docs/apendice-pruebas.md](docs/apendice-pruebas.md) | Para no técnicos: cómo se prueba el modelo y cuatro ejemplos al azar |
| [docs/correr-el-modelo.md](docs/correr-el-modelo.md) | Usar el escritor: línea de comandos, app, Python, `mlx_lm`, problemas comunes |
| [docs/entrenar-y-evaluar.md](docs/entrenar-y-evaluar.md) | Reproducir el modelo final: corpus, GRPO, selección + SFT, Inspect |
| [GLOSSARY.md](GLOSSARY.md) | El lenguaje del dominio: fragmento, imaginación, ganancia, juez… |
| [docs/adr/](docs/adr/) | Decisiones de diseño difíciles de revertir |
| [CONSTITUTION.md](CONSTITUTION.md) | Principios de diseño del código |
| [AGENTS.md](AGENTS.md) | Instrucciones para agentes y verificación antes de commitear |

## La receta de DeepSeek-R1, aplicada a la escritura

Adaptación de la receta de [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (RL puro con GRPO y recompensas basadas en reglas sobre un modelo base) a la escritura literaria en lugar de la matemática.

| DeepSeek-R1-Zero | imagine |
|---|---|
| DeepSeek-V3-Base (671B) | [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) (feb. 2026, sin instrucciones ni razonamiento) |
| `<think> … </think>` antes de responder | `<imagine> … </imagine>` antes de continuar la novela |
| Recompensa de exactitud: ¿la respuesta es correcta? | Recompensa de corpus: cuánto mejora lo imaginado la predicción de lo que el autor escribió de verdad |
| Recompensa de formato | Recompensa de formato: cierra `</imagine>`, no está vacío, no reabre la etiqueta |
| GRPO | GRPO (objetivo recortado + KL k3 contra el modelo base), con LoRA sobre MLX |

Después del RL viene, como en R1, una etapa de selección y ajuste supervisado. El resultado es un escritor que imagina la escena y después continúa la novela.

## Recompensa de corpus

Cada ejemplo es un fragmento de una novela en español (Project Gutenberg) y su referencia: lo que el autor escribió a continuación, que el escritor nunca ve. El juez, el modelo base congelado, mide la **ganancia**:

```
Δ = log p(referencia | fragmento, imaginación) − log p(referencia | fragmento, imaginación vacía)   (media por token)
recompensa = 1 + tanh(4·Δ)   si el formato es válido, 0 si no
```

La imaginación no ve la referencia, así que sólo suma si anticipa lo que de verdad pasa. Se verifica contra el corpus sin entrenar un modelo de recompensa, por la misma razón por la que R1 evita las recompensas neuronales.

- Entrenamiento: [*Don Quijote*](https://www.gutenberg.org/ebooks/2000) (Cervantes) y [*Fortunata y Jacinta*](https://www.gutenberg.org/ebooks/17013) (Pérez Galdós).
- Prueba: [*Niebla*](https://www.gutenberg.org/ebooks/49836) (Unamuno). Ni el libro ni su autor aparecen en el entrenamiento.

Una diferencia con R1-Zero: la plantilla incluye un ejemplo (one-shot, [ADR 0002](docs/adr/0002-plantilla-con-un-ejemplo.md)). Sin él, el modelo de 0.8B cierra la etiqueta sólo ~5 % de las veces; con él, ~90 % a temperatura 0.8.

## Resultados

300 pasos de GRPO (4 fragmentos × G=8 por paso, lr 2e-6, LoRA rango 16), ~3 h en una M2 Max, con un checkpoint cada 50 pasos. Cada modelo se evaluó sobre 40 fragmentos de *Niebla* con 4 intentos por fragmento, dos veces: con el script `evaluar` original (hoy reemplazado por la tarea `imagine/niebla`) y con Inspect (`evals/niebla_flow.py`). La columna Δ está en nats/token.

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
- La mejora crece hasta el paso ~150 y después se estanca. Entre los pasos 150 y 300 las diferencias (~0,005) están dentro del ruido: el error estándar de Δ es ±0,003 por intento, y es mayor porque los 4 intentos de un fragmento están correlacionados.
- Las dos evaluaciones usan intentos distintos y coinciden dentro de ese ruido.
- La prosa sigue siendo la de un modelo de 0.8B.

## Segunda etapa: selección + SFT

Como en DeepSeek-R1, después del RL viene una etapa supervisada (`imagine.entrenamiento.sft`; en el paper, *rejection sampling*):

1. El modelo de RL (paso 250) imagina 8 veces sobre 600 fragmentos de entrenamiento. Por cada fragmento, la selección se queda con la imaginación de mayor ganancia, si ayuda (Δ > 0). Quedan 569 ejemplos supervisados.
2. Ajuste supervisado sobre fragmento → imaginación elegida → **referencia**. El RL sólo premiaba la imaginación; acá el modelo aprende también a escribir la continuación como Cervantes o Galdós.

La pérdida de validación baja de 3,33 a 3,03 en la época 1 y sube en la 2, así que el modelo final es `runs/sft/epoca_1`.

### Evaluación final en *Niebla* (Inspect)

Imaginación (tarea `imagine/niebla`, 40 fragmentos × 4 intentos; Δ en nats/token):

| Modelo | Formato | Δ | Imaginaciones que ayudan |
|---|---|---|---|
| base | 84 % | −0,026 | 28 % |
| RL paso 150 | 94 % | +0,026 | 75 % |
| RL paso 300 | 97 % | +0,021 | 67 % |
| RL + SFT | 99 % | +0,030 | 77 % |

Prosa de la continuación (tarea `imagine/prosa`, 40 fragmentos × 2 intentos):

| Modelo | Repetición ↓ | Diversidad ↑ | Fluidez ↑ | Parecido con el autor ↑ | Palabras |
|---|---|---|---|---|---|
| base, sin imaginar | 0,016 | 0,968 | −2,20 | 0,156 | 39 |
| base, imaginando | 0,021 | 0,951 | −2,08 | 0,135 | 32 |
| RL paso 150 | 0,006 | 0,969 | −3,55 | 0,203 | 54 |
| RL paso 300 | 0,002 | 0,982 | −3,62 | 0,215 | 64 |
| RL + SFT | 0,011 | 0,967 | −2,87 | 0,169 | 38 |

- El SFT mantiene (y mejora un poco) la imaginación del RL.
- El SFT recupera parte de la fluidez que perdió el RL y escribe continuaciones del largo de las del autor. El parecido alto del RL viene en parte de que escribe más.
- Con un largo similar al del modelo base sin imaginar, el SFT se parece más al autor (0,169 contra 0,156).
- La fluidez se mide sin la imaginación en el contexto, así que penaliza continuaciones que siguen lo imaginado. La prosa sigue siendo la de un modelo de 0.8B.

## Lenguaje del dominio

El glosario del proyecto está en [GLOSSARY.md](GLOSSARY.md): fragmento, imaginación, continuación, referencia, ganancia, recompensa, selección, etc. El código usa esos mismos términos. Las decisiones difíciles de revertir están en [docs/adr/](docs/adr/):

- [0001](docs/adr/0001-recompensa-por-ganancia-con-modelo-base.md): la recompensa es la ganancia medida por el modelo base congelado.
- [0002](docs/adr/0002-plantilla-con-un-ejemplo.md): la plantilla incluye un ejemplo, a diferencia de R1-Zero.
- [0003](docs/adr/0003-gradiente-solo-por-lo-generado.md): el gradiente pasa sólo por lo que genera el escritor.
- [0004](docs/adr/0004-mlx-en-apple-silicon.md): se entrena y se sirve con MLX en Apple Silicon.

Lo que en logs y tablas viejas figura como `delta` es la **ganancia** (Δ). Desde el modelado del dominio, los logs nuevos de GRPO y las métricas de `imagine/niebla` la llaman `ganancia`. El formato de los jsonl de `data/` no cambió. En la tarea `imagine/prosa`, `chrf` pasó a llamarse `parecido` y `ajenos`, `letras_ajenas`; las tareas reciben `intentos` en lugar de `muestras`, y el log de GRPO guarda la `mejor_imaginacion` del grupo en lugar de `muestra`.

## Arquitectura

El código sigue los principios de [CONSTITUTION.md](CONSTITUTION.md). Cada paquete de `imagine/` tiene una sola responsabilidad, y las dependencias van de arriba hacia abajo:

| Paquete | Responsabilidad | Depende de |
|---|---|---|
| `dominio` | Modelo del dominio (Python puro): corpus y libros, plantillas, imaginación, ejemplo y ejemplo supervisado, recompensa, selección, historia a cuatro manos, métricas de prosa | — |
| `datos` | Corpus de Gutenberg; repositorio de ejemplos en jsonl | `dominio` |
| `persistencia` | Guardar y leer adaptadores LoRA; catálogo de los modelos entrenados | — |
| `ejecucion` | MLX: carga con caché única, hilo dedicado, muestreo, escritor, puntuación contra el corpus | `dominio`, `datos`, `persistencia` |
| `entrenamiento` | GRPO, selección + SFT, LoRA | `ejecucion` y anteriores |
| `integraciones.inspect` | Proveedor `mlx/` y tareas `imagine/niebla` e `imagine/prosa` | `ejecucion` y anteriores |
| `interfaz` | El comando `imagine`: app de Gradio, historia a cuatro manos y fragmento en la terminal | `ejecucion` y anteriores |

Donde el código pide un modelo, se acepta un repo de Hugging Face (`Qwen/Qwen3.5-0.8B-Base`) o una carpeta de adaptadores del proyecto (`runs/sft/epoca_1`). Los modelos entrenados de referencia (el RL y el final) están en `imagine/persistencia/catalogo.py`. Los presupuestos de generación (tokens y temperaturas) están en `imagine/dominio/escritura.py`.

`tests/test_arquitectura.py` verifica que cada capa importe sólo las de abajo y que el dominio no dependa de MLX, Inspect ni Gradio.

## Apéndice

[El modelo, puesto a prueba](docs/apendice-pruebas.md): una explicación para gente no técnica y cuatro pruebas reales en *Niebla*, con el modelo sin entrenar y el entrenado.
