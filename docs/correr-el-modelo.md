# Correr el modelo

Todo se usa con un solo comando, `imagine`:

| Comando | Qué hace |
|---|---|
| `uv run imagine app` | Abre la app en el navegador |
| `uv run imagine cuatro-manos` | Escribe una historia por turnos en la terminal |
| `uv run imagine generar "…"` | Imagina y continúa un fragmento, y termina |
| `uv run imagine plantillas` | Lista las plantillas para presentarle la escena |

## Instalar

Hace falta una Mac con chip Apple (M1 o posterior). No corre en Windows, Linux ni Macs con Intel ([ADR 0004](adr/0004-mlx-en-apple-silicon.md)).

1. Instalá [uv](https://docs.astral.sh/uv/), que se encarga de Python y las dependencias, y [Git LFS](https://git-lfs.com/), que baja los modelos:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   brew install git-lfs && git lfs install
   ```

2. Cloná el repo. Con Git LFS instalado, el modelo entrenado viene incluido:

   ```bash
   git clone https://github.com/andrestobelem/train.git && cd train
   ```

3. Instalá las dependencias:

   ```bash
   uv sync
   ```

La primera vez que se usa, también se baja el modelo base de Hugging Face (~1,7 GB). Después funciona sin conexión.

Si clonaste sin Git LFS o falta el modelo entrenado, `imagine` lo avisa y dice qué correr. El modelo final también está en Hugging Face: `uv run hf download andrestobelem/imagine-qwen3.5-0.8b --local-dir runs/sft/epoca_1`.

## La app

```bash
uv run imagine app
```

Se abre el navegador en http://127.0.0.1:7860. Para cerrarla, Ctrl-C en la terminal. Al pasar el mouse sobre cada control, aparece qué hace. Tiene dos pestañas:

- **Fragmento.** Pegás unas líneas de una novela y apretás «Continuar». A la derecha aparecen la imaginación y, debajo, la continuación. Abajo hay fragmentos de *Niebla* para probar sin escribir nada; aparecen si antes se armó el corpus con `uv run python -m imagine.datos.corpus`.
- **A cuatro manos.** Escribís un párrafo, el modelo escribe el siguiente y vuelve a ser tu turno. La imaginación de cada turno aparece plegada arriba de su respuesta.

En las dos podés elegir la [plantilla](#plantillas), la temperatura, el largo y el modelo. Los modelos son:

- **Entrenado (final)**: el de este proyecto.
- **Sin entrenar (base)**: para comparar.
- **Sólo RL (paso 250)**: aparece sólo en la máquina donde se entrenó.

Opciones: `--puerto 7861` para usar otro puerto y `--no-abrir` para no abrir el navegador.

## En la terminal

### A cuatro manos

```bash
uv run imagine cuatro-manos
```

```
Lector> Llovía.

<imagine> Quiere decirse que ella estaba muy preparada para el desmayo, porque había tenido la sensación de estar fuerte y de haberse de una manera. </imagine>

Escritor> — Él no es el de la calle. ¿Y él es el de la calle?

Lector>
```

- Escribí un párrafo y apretá Enter.
- Con una línea vacía pasás el turno y el modelo sigue solo.
- Para terminar: `/fin`, Ctrl-D o Ctrl-C.

### Un fragmento

```bash
uv run imagine generar "Augusto salió a la calle sin saber adónde ir. Llovía."
uv run imagine generar < fragmento.txt
```

Imprime la imaginación y la continuación, y termina. Cada vez carga el modelo de nuevo (unos segundos); para probar muchos fragmentos conviene la app.

### Opciones

| Opción | Dónde | Por defecto | Para qué |
|---|---|---|---|
| `--modelo` | las dos | el entrenado | `Qwen/Qwen3.5-0.8B-Base` para comparar, u otra carpeta de `runs/` |
| `--temperatura` | las dos | 0,7 | Cuánto se arriesga al elegir cada palabra: en 0 es previsible y se repite; por encima de 1 sorprende más y pierde coherencia |
| `--max-tokens` | las dos | 160 por turno, 200 en `generar` | Largo máximo de lo que escribe. Un token es un pedazo de palabra: 100 tokens son unas 65 palabras |
| `--plantilla` | las dos | `conversación` | Cómo se le presenta la escena (ver [Plantillas](#plantillas)) |
| `--ocultar-imaginacion` | `cuatro-manos` | — | No muestra lo imaginado |
| `--sin-imaginar` | `generar` | — | Continúa sin imaginar antes, para comparar |

`uv run imagine <comando> --help` lista las opciones de cada uno.

## Plantillas

La plantilla es el texto con el que se le presenta la escena al modelo: una consigna, un ejemplo resuelto y el fragmento. Cada plantilla pide imaginar de otra manera:

| Plantilla | Qué imagina el modelo antes de escribir |
|---|---|
| `conversación` | Un lector comparte un fragmento y el escritor lo continúa. **Es con la que se entrenó** y la que se usa por defecto. |
| `novela` | Lo que viene, sin roles ni conversación |
| `imaginar` | La escena con los sentidos: el lugar, los sonidos, los objetos |
| `soñar` | Un sueño de cómo sigue |
| `leer` | Lo que va a pasar, como un lector antes de dar vuelta la página |
| `escribir` | Qué tiene que lograr el párrafo siguiente: qué revelar, qué callar |
| `cuatro-manos` | Cómo sigue, como uno de dos autores que se turnan |
| `taller` | El tono y lo que sienten los personajes, para que no se note el cambio de mano |
| `folletín` | Qué pasa en la próxima entrega |
| `teatro` | La puesta: dónde está cada uno, los gestos, los silencios |
| `personaje` | Lo que siente el personaje, en primera persona |
| `recordar` | Cómo fue de verdad, como una autora que escribe lo que vivió |

Todas usan el mismo ejemplo (Elena y la carta), así que lo único que cambia es el encuadre. Sólo la de conversación sigue el ejemplo con un diálogo; las demás siguen con narración.

El modelo **se entrenó sólo con `conversación`**. Las demás andan sin reentrenar, pero el modelo no las aprendió: sirven para explorar y comparar.

```bash
uv run imagine plantillas           # la lista
uv run imagine plantillas soñar     # el texto completo de una
uv run imagine generar --plantilla soñar "Augusto salió a la calle sin saber adónde ir."
uv run imagine cuatro-manos --plantilla cuatro-manos
```

El nombre se puede escribir sin tildes (`sonar`, `folletin`). Las plantillas están en [`imagine/dominio/plantillas.py`](../imagine/dominio/plantillas.py). Para evaluar con otra, las tareas de Inspect aceptan `-T plantilla=soñar`.

## Qué esperar

Es un modelo chico (800 millones de parámetros) que continúa texto: **no responde preguntas ni sigue instrucciones**. Si le escribís «contame un cuento», no lo cuenta: sigue esa frase como si fuera parte de una novela.

- Funciona mejor con prosa narrativa en español, de unas pocas líneas hasta 180 palabras. Si le das más, ve sólo las últimas 180.
- Escribe frases correctas con tono de novela del siglo XIX, pero pierde el hilo en pocas líneas.
- Sus imaginaciones son vagas y repiten giros como «se había de». El [apéndice](apendice-pruebas.md) explica por qué y muestra ejemplos.

## Para programadores

### Qué es el modelo

Un modelo de este proyecto son **adaptadores LoRA** (~40 MB) que se suman al [modelo base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) al cargarlo. Cada carpeta tiene `adapters.safetensors` y un `adapter_config.json` que dice cuál es su modelo base. Las rutas de los modelos entrenados están en [`imagine/persistencia/catalogo.py`](../imagine/persistencia/catalogo.py).

### Desde Python

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

MLX ata sus streams al hilo que los creó. Desde un servidor con varios hilos, como Gradio o Inspect, corré todo lo que toque al modelo dentro de `imagine.ejecucion.hilo.en_hilo_mlx`.

### Con `mlx_lm`

Los adaptadores también andan con `mlx_lm` directamente, si se le da la plantilla con la que se entrenó (`PLANTILLA` en [`imagine/dominio/escritura.py`](../imagine/dominio/escritura.py)) y se desactiva la plantilla de chat:

```bash
uv run python -m mlx_lm generate \
  --model Qwen/Qwen3.5-0.8B-Base --adapter-path runs/sft/epoca_1 \
  --ignore-chat-template --temp 0.7 --max-tokens 200 \
  --prompt "$(uv run python -c 'from imagine.dominio.escritura import prompt; print(prompt("Augusto salió a la calle…"), end="")')"
```

`mlx_lm` no sabe dónde termina un turno, así que puede seguir escribiendo `Lector:` y otro fragmento. `mlx_lm chat` no sirve: envuelve cada mensaje en la plantilla de chat de Qwen y el modelo comenta el texto en lugar de continuarlo.

## Problemas comunes

| Síntoma | Qué hacer |
|---|---|
| «Los modelos vienen con el repo por Git LFS, pero no se bajaron» | Clonaste sin Git LFS. Corré `git lfs install` y `git lfs pull` en la carpeta del repo. |
| `401` o `403` al bajar de Hugging Face | Ese repo es privado: hace falta iniciar sesión con `uv run hf auth login` y tener acceso. Con Git LFS no hace falta. |
| `CERTIFICATE_VERIFY_FAILED` al bajar algo | Un proxy de la red intercepta la conexión. Bajá el modelo desde otra red o configurá `REQUESTS_CA_BUNDLE`. |
| «El puerto 7860 está ocupado» al abrir la app | Ya hay una app abierta: usala, cerrala o abrí otra con `--puerto 7861`. |
| `There is no Stream(cpu, 0) in current thread` | Se usó MLX desde otro hilo. Envolvé la llamada con `en_hilo_mlx`. |
