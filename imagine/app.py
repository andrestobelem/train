"""App local de Gradio para probar los modelos: imagina y después continúa un fragmento.

    uv run python -m imagine.app      # http://127.0.0.1:7860
"""

import json
from pathlib import Path

import gradio as gr

from .generar import imaginar_y_continuar
from .modelo import cargar, en_hilo_mlx

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = {
    "RL + SFT (final)": "runs/sft/epoca_1",
    "RL (GRPO, paso 250)": "runs/imagine300/paso_00250",
    "Base (Qwen3.5-0.8B-Base)": None,
}
_cargados = {}


def _modelo(nombre: str):
    """Carga cada modelo la primera vez que se pide (~1,6 GB cada uno)."""
    if nombre not in _cargados:
        adaptadores = MODELOS[nombre]
        _cargados[nombre] = cargar(str(RAIZ / adaptadores) if adaptadores else None)
    return _cargados[nombre]


def responder(fragmento: str, nombre: str, imaginar: bool, temperatura: float, largo: int):
    if not fragmento.strip():
        return "", ""

    def correr():
        model, tok = _modelo(nombre)
        return imaginar_y_continuar(model, tok, fragmento, temperatura, largo, imaginar)

    # Gradio atiende en sus propios hilos; MLX tiene que correr siempre en el mismo.
    return en_hilo_mlx(correr)


def ejemplos(n: int = 4) -> list[list[str]]:
    """Fragmentos de Niebla, el libro que el modelo no vio al entrenar."""
    ruta = RAIZ / "data" / "test.jsonl"
    if not ruta.exists():
        return []
    filas = [json.loads(l) for l in open(ruta, encoding="utf-8")][:n]
    return [[f["contexto"]] for f in filas]


with gr.Blocks(title="imagine") as demo:
    gr.Markdown(
        "# imagine\n"
        "Qwen3.5-0.8B-Base entrenado a la DeepSeek-R1 para **imaginar** antes de "
        "continuar una novela. Pegá un fragmento en español."
    )
    with gr.Row():
        with gr.Column():
            fragmento = gr.Textbox(label="Fragmento", lines=10)
            modelo = gr.Radio(list(MODELOS), value=list(MODELOS)[0], label="Modelo")
            imaginar = gr.Checkbox(value=True, label="Imaginar antes de escribir")
            temperatura = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperatura")
            largo = gr.Slider(32, 400, value=200, step=8, label="Tokens de la continuación")
            boton = gr.Button("Continuar", variant="primary")
        with gr.Column():
            imaginado = gr.Textbox(label="<imagine>", lines=5)
            continuacion = gr.Textbox(label="Continuación", lines=12)
    if ejemplos():
        gr.Examples(ejemplos(), inputs=[fragmento], label="Fragmentos de Niebla")
    boton.click(
        responder,
        inputs=[fragmento, modelo, imaginar, temperatura, largo],
        outputs=[imaginado, continuacion],
    )


if __name__ == "__main__":
    demo.launch()
