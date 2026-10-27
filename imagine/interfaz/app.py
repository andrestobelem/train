"""App local de Gradio para probar los modelos: imagina y después continúa un fragmento.

uv run python -m imagine.interfaz.app      # http://127.0.0.1:7860
"""

import gradio as gr

from imagine.datos.archivos import TEST, leer_jsonl
from imagine.dominio.escritura import TEMPERATURA_ESCRITURA, TOKENS_CONTINUACION
from imagine.ejecucion.escritor import imaginar_y_continuar
from imagine.ejecucion.hilo import en_hilo_mlx
from imagine.ejecucion.modelos import MODELO_BASE, cargar_por_nombre

MODELOS = {
    "RL + SFT (final)": "runs/sft/epoca_1",
    "RL (GRPO, paso 250)": "runs/imagine300/paso_00250",
    "Base (Qwen3.5-0.8B-Base)": MODELO_BASE,
}


def responder(fragmento: str, etiqueta: str, imaginar: bool, temperatura: float, largo: int):
    if not fragmento.strip():
        return "", ""

    def correr():
        model, tok = cargar_por_nombre(MODELOS[etiqueta])
        return imaginar_y_continuar(model, tok, fragmento, temperatura, largo, imaginar)

    # Gradio atiende en sus propios hilos; MLX tiene que correr siempre en el mismo.
    return en_hilo_mlx(correr)


def ejemplos(n: int = 4) -> list[list[str]]:
    """Fragmentos de Niebla, el libro que el modelo no vio al entrenar."""
    if not TEST.exists():
        return []
    return [[f["contexto"]] for f in leer_jsonl(TEST, n)]


def construir() -> gr.Blocks:
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
                temperatura = gr.Slider(
                    0.0, 1.5, value=TEMPERATURA_ESCRITURA, step=0.05, label="Temperatura"
                )
                largo = gr.Slider(
                    32, 400, value=TOKENS_CONTINUACION, step=8, label="Tokens de la continuación"
                )
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
    return demo


if __name__ == "__main__":
    construir().launch()
