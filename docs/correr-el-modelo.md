# Correr el modelo

Cómo usar el escritor entrenado para que imagine y continúe un fragmento: desde la línea de comandos, con la app de Gradio, desde Python o con `mlx_lm` directamente.

## Requisitos

- Una Mac con Apple Silicon (M1 o posterior). El proyecto usa [MLX](https://github.com/ml-explore/mlx) y no corre en Linux, Windows ni Macs con Intel (ver [ADR 0004](adr/0004-mlx-en-apple-silicon.md)).
- [uv](https://docs.astral.sh/uv/). Instala por su cuenta Python 3.12 o posterior.
- Para usar el modelo alcanzan unos 3 GB de memoria; generar un fragmento usa ~2,3 GB. Entrenar es otra cosa (ver [entrenar-y-evaluar.md](entrenar-y-evaluar.md)).

```bash
uv sync
```

La primera vez que se carga un modelo, `mlx_lm` descarga el [modelo base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) de Hugging Face (~1,7 GB) y lo deja en la caché de `~/.cache/huggingface`. Después funciona sin conexión.

## Qué modelo se usa

Un modelo entrenado por este proyecto no es una copia del modelo base: son **adaptadores LoRA** (~40 MB) que se suman al modelo base al cargarlo. Cada carpeta de adaptadores tiene `adapters.safetensors` y un `adapter_config.json` que dice cuál es su modelo base.

| Nombre | Carpeta | Qué es |
|---|---|---|
| Modelo final | `runs/sft/epoca_1` | RL + selección + ajuste supervisado. Es el que se usa por defecto. |
| RL | `runs/imagine300/paso_00250` | Sólo GRPO, paso 250. Imagina bien y escribe más largo y menos fluido. |
| Modelo base | `Qwen/Qwen3.5-0.8B-Base` | Sin entrenar por este proyecto, para comparar. |

Esas rutas están en [`imagine/persistencia/catalogo.py`](../imagine/persistencia/catalogo.py). Donde el código pide un modelo, acepta un repo de Hugging Face o una carpeta de adaptadores.

### Bajar el modelo final

`runs/` no está en git, así que al clonar el repositorio los adaptadores no vienen incluidos. El modelo final está en Hugging Face, en el repo privado [andrestobelem/imagine-qwen3.5-0.8b](https://huggingface.co/andrestobelem/imagine-qwen3.5-0.8b). Para bajarlo necesitás una cuenta con acceso a ese repo:

```bash
uv run hf auth login
uv run hf download andrestobelem/imagine-qwen3.5-0.8b --local-dir runs/sft/epoca_1
```

Se baja en la carpeta que el código usa por defecto, así que después alcanza con `uv run python -m imagine.interfaz.generar "…"`. Si no tenés acceso, pedíselo al dueño del repo. También podés entrenar los adaptadores siguiendo [entrenar-y-evaluar.md](entrenar-y-evaluar.md), o usar sólo el modelo base con `--modelo Qwen/Qwen3.5-0.8B-Base`.

El checkpoint de RL (`runs/imagine300/paso_00250`) no está publicado: para usarlo, entrenalo.

## Qué esperar

Es un modelo base de 0.8B: **no sigue instrucciones ni conversa**. Recibe un fragmento de novela en español y hace dos cosas:

1. Imagina brevemente la escena entre `<imagine>` y `</imagine>`: qué sienten los personajes y cómo sigue.
2. Escribe la continuación, que termina cuando le tocaría el turno al lector.

Funciona mejor con prosa narrativa en español, de unas 50 a 180 palabras. Si el texto es más largo, el escritor ve sólo las últimas 180 palabras. La prosa es la de un modelo de 0.8B: suele ser coherente frase a frase, pero pierde el hilo rápido.

## Línea de comandos

```bash
uv run python -m imagine.interfaz.generar "Augusto salió a la calle sin saber adónde ir. Llovía, y el paraguas abierto le parecía un techo que lo seguía."
```

```
<imagine> Augusto se había de armar con el paraguas, porque no quería ser despojado por la lluvia; … </imagine>

— ¿Qué te viene? —dijo un joven que le miraba a él desde la esquina.
…
```

| Opción | Por defecto | Para qué |
|---|---|---|
| `--modelo` | `runs/sft/epoca_1` | Otra carpeta de adaptadores o un repo de Hugging Face |
| `--temperatura` | 0,7 | Más alta, más variada y menos coherente |
| `--max-tokens` | 200 | Largo máximo de la continuación (la imaginación tiene su propio tope de 96 tokens) |
| `--sin-imaginar` | — | Continúa sin imaginar antes, para comparar |

Cada corrida carga el modelo de nuevo (~2 s). Para probar muchos fragmentos conviene la app.

## A cuatro manos en la terminal

```bash
uv run python -m imagine.interfaz.cuatro_manos
```

Es la historia a cuatro manos de la app, pero en la terminal. Escribís un párrafo como lector y apretás Enter; el escritor imagina (en gris) y sigue la historia, y vuelve a ser tu turno. Una línea vacía pasa el turno, así que el escritor sigue solo. `/fin` o Ctrl-D terminan. Acepta `--modelo`, `--temperatura`, `--max-tokens` (160 por turno) y `--sin-imaginacion` para no mostrar lo imaginado.

> [!NOTE]
> No sirve `mlx_lm chat`: envuelve cada mensaje en la plantilla de chat de Qwen, y el escritor se entrenó con su propia plantilla de lector y escritor. Con la plantilla de chat, el modelo comenta el texto en lugar de imaginar y continuar.

## App de Gradio

```bash
uv run python -m imagine.interfaz.app
```

Abrí http://127.0.0.1:7860. La app carga cada modelo una vez y lo mantiene en memoria. Tiene dos pestañas:

- **Fragmento**: pegás un fragmento, elegís el modelo y ves por separado la imaginación y la continuación. Podés apagar «Imaginar antes de escribir» para comparar. Si existe `data/test.jsonl`, abajo aparecen fragmentos de *Niebla* como ejemplos; ese archivo se genera con `uv run python -m imagine.datos.corpus`.
- **A cuatro manos**: escribís un párrafo, el escritor sigue la historia y vuelve a ser tu turno. La imaginación de cada turno aparece plegada y no forma parte de la historia.

La app sólo escucha en la máquina local. Si necesitás un enlace público temporal, cambiá `construir().launch()` por `construir().launch(share=True)` en `imagine/interfaz/app.py`.

## Desde Python

```python
from imagine.ejecucion.escritor import imaginar_y_continuar
from imagine.ejecucion.modelos import cargar_por_nombre
from imagine.persistencia import catalogo

model, tok = cargar_por_nombre(catalogo.FINAL)  # se carga una vez por proceso
imaginacion, continuacion = imaginar_y_continuar(model, tok, "Augusto salió a la calle…")
print(imaginacion.texto, imaginacion.formato_valido)
print(continuacion)
```

Para escribir por turnos, `turno_del_escritor` recibe una `Historia` (de `imagine.dominio.historia`) y la devuelve con el turno del escritor agregado.

> [!NOTE]
> MLX ata sus streams al hilo que los creó. Si lo usás desde un servidor con varios hilos, como Gradio o Inspect, corré todo lo que toque al modelo dentro de `imagine.ejecucion.hilo.en_hilo_mlx`.

## Con `mlx_lm` directamente

Los adaptadores son LoRA estándar de `mlx_lm`, así que también se pueden usar sin este paquete. Hay que darle al modelo la plantilla con la que se entrenó (`PLANTILLA` en [`imagine/dominio/escritura.py`](../imagine/dominio/escritura.py)) y desactivar la plantilla de chat:

```bash
uv run python -m mlx_lm generate \
  --model Qwen/Qwen3.5-0.8B-Base --adapter-path runs/sft/epoca_1 \
  --ignore-chat-template --temp 0.7 --max-tokens 200 \
  --prompt "$(uv run python -c 'from imagine.dominio.escritura import prompt; print(prompt("Augusto salió a la calle…"), end="")')"
```

`mlx_lm` no sabe dónde termina un turno, así que puede seguir escribiendo `Lector:` y otro fragmento. La imaginación y la continuación salen en un solo paso y a la misma temperatura. El código del proyecto, en cambio, corta la imaginación en `</imagine>` y la continuación en el turno siguiente.

## Problemas comunes

| Síntoma | Causa y solución |
|---|---|
| `RepositoryNotFoundError` (401) con `runs/...` | La carpeta de adaptadores no existe y el nombre se interpretó como un repo de Hugging Face. Revisá que esté `adapter_config.json` dentro de la carpeta, o bajala como se explica en [Bajar el modelo final](#bajar-el-modelo-final). |
| `401` o `403` al bajar el modelo final | No iniciaste sesión, o tu cuenta no tiene acceso al repo privado. Iniciá sesión con `uv run hf auth login` y pedí acceso. |
| `There is no Stream(cpu, 0) in current thread` | Se usó MLX desde otro hilo. Envolvé la llamada con `en_hilo_mlx`. |
| Falla la descarga con `CERTIFICATE_VERIFY_FAILED` | Un proxy corporativo intercepta TLS. Para el corpus el proyecto ya usa los certificados del sistema. Para Hugging Face, descargá el modelo una vez desde una red sin proxy o configurá `REQUESTS_CA_BUNDLE`. |
| La imaginación sale cortada o sin `</imagine>` | Llegó al tope de 96 tokens sin cerrarla. Pasa poco con el modelo final y más con el base o a temperatura alta. El escritor la cierra y continúa igual, pero esa imaginación no tiene formato válido. |
