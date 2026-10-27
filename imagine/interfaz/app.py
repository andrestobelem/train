"""App local de Gradio para probar los modelos: imagina y después continúa un fragmento.

uv run python -m imagine.interfaz.app      # http://127.0.0.1:7860
"""

import gradio as gr

from imagine.datos.archivos import TEST
from imagine.datos.repositorio import ejemplos_de_prueba
from imagine.dominio.escritura import TEMPERATURA_ESCRITURA, TOKENS_CONTINUACION
from imagine.ejecucion.escritor import imaginar_y_continuar
from imagine.ejecucion.hilo import en_hilo_mlx
from imagine.ejecucion.modelos import MODELO_BASE, cargar_por_nombre
from imagine.persistencia import catalogo

MODELOS = {
    "RL + SFT (final)": catalogo.FINAL,
    "RL (GRPO, paso 250)": catalogo.RL,
    "Base (Qwen3.5-0.8B-Base)": MODELO_BASE,
}


def responder(fragmento: str, etiqueta: str, imaginar: bool, temperatura: float, largo: int):
    if not fragmento.strip():
        return "", ""

    def correr():
        model, tok = cargar_por_nombre(MODELOS[etiqueta])
        imaginacion, continuacion = imaginar_y_continuar(
            model, tok, fragmento, temperatura, largo, imaginar
        )
        return imaginacion.texto.strip(), continuacion

    # Gradio atiende en sus propios hilos; MLX tiene que correr siempre en el mismo.
    return en_hilo_mlx(correr)


def ejemplos(n: int = 4) -> list[list[str]]:
    """Fragmentos del libro de prueba (Niebla), que el escritor no vio al entrenar."""
    if not TEST.exists():
        return []
    return [[ejemplo.fragmento] for ejemplo in ejemplos_de_prueba(n)]


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
                imaginacion = gr.Textbox(label="<imagine>", lines=5)
                continuacion = gr.Textbox(label="Continuación", lines=12)
        if ejemplos():
            gr.Examples(ejemplos(), inputs=[fragmento], label="Fragmentos de Niebla")
        boton.click(
            responder,
            inputs=[fragmento, modelo, imaginar, temperatura, largo],
            outputs=[imaginacion, continuacion],
        )
    return demo


if __name__ == "__main__":
    construir().launch()
