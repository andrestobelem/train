"""App local de Gradio para probar los modelos.

- Fragmento: el escritor imagina y después continúa un fragmento.
- A cuatro manos: el lector y el escritor se turnan para escribir una historia.

uv run imagine app      # abre http://127.0.0.1:7860 en el navegador
"""

import argparse
import sys

import gradio as gr

from imagine.datos.archivos import TEST
from imagine.datos.repositorio import ejemplos_de_prueba
from imagine.dominio.escritura import TEMPERATURA_ESCRITURA, TOKENS_CONTINUACION
from imagine.dominio.historia import Historia, Rol, Turno
from imagine.ejecucion.escritor import imaginar_y_continuar, turno_del_escritor
from imagine.ejecucion.hilo import en_hilo_mlx
from imagine.ejecucion.modelos import MODELO_BASE, cargar_por_nombre
from imagine.persistencia import catalogo

from .modelo import disponible, falta

TODOS = {
    "Entrenado (final)": catalogo.FINAL,
    "Sólo RL (paso 250)": catalogo.RL,
    "Sin entrenar (base)": MODELO_BASE,
}
# Sólo los que están en esta máquina: el de RL, por ejemplo, no se publica.
MODELOS = {etiqueta: nombre for etiqueta, nombre in TODOS.items() if disponible(nombre)}


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


# Título de los mensajes del chat que muestran la imaginación: no son parte de la historia.
TITULO_IMAGINACION = "<imagine>"


def _texto(mensaje: dict) -> str:
    """Texto de un mensaje del historial de Gradio (string o lista de partes)."""
    contenido = mensaje["content"]
    if isinstance(contenido, str):
        return contenido
    return "".join(p.get("text", "") for p in contenido if isinstance(p, dict))


def historia_desde_chat(historial: list[dict], mensaje: str) -> Historia:
    """Traduce el chat de Gradio a la historia: el usuario es el lector y el asistente,
    el escritor; los mensajes con la imaginación no son turnos."""
    historia = Historia()
    for m in historial:
        if (m.get("metadata") or {}).get("title") == TITULO_IMAGINACION:
            continue
        rol = Rol.LECTOR if m["role"] == "user" else Rol.ESCRITOR
        historia = historia.agregar(Turno(rol, _texto(m)))
    return historia.agregar(Turno(Rol.LECTOR, mensaje))


def escribir_a_cuatro_manos(mensaje, historial, etiqueta, temperatura, largo):
    historia = historia_desde_chat(historial, mensaje)

    def correr():
        model, tok = cargar_por_nombre(MODELOS[etiqueta])
        return turno_del_escritor(model, tok, historia, temperatura, largo)

    imaginacion, historia = en_hilo_mlx(correr)
    return [
        gr.ChatMessage(content=imaginacion.texto.strip(), metadata={"title": TITULO_IMAGINACION}),
        gr.ChatMessage(content=historia.turnos[-1].texto),
    ]


def ejemplos(n: int = 4) -> list[list[str]]:
    """Fragmentos del libro de prueba (Niebla), que el escritor no vio al entrenar."""
    if not TEST.exists():
        return []
    return [[ejemplo.fragmento] for ejemplo in ejemplos_de_prueba(n)]


def construir() -> gr.Blocks:
    with gr.Blocks(title="imagine") as demo:
        gr.Markdown(
            "# imagine\n"
            "Un modelo chico que **imagina** la escena antes de continuar una novela en "
            "español. No responde preguntas ni sigue instrucciones: continúa el texto que "
            "le des."
        )
        with gr.Tab("Fragmento"):
            with gr.Row():
                with gr.Column():
                    fragmento = gr.Textbox(
                        label="Fragmento",
                        lines=10,
                        placeholder="Pegá unas líneas de una novela y apretá Continuar.",
                    )
                    modelo = gr.Radio(list(MODELOS), value=list(MODELOS)[0], label="Modelo")
                    imaginar = gr.Checkbox(value=True, label="Imaginar antes de escribir")
                    temperatura = gr.Slider(
                        0.0, 1.5, value=TEMPERATURA_ESCRITURA, step=0.05, label="Temperatura"
                    )
                    largo = gr.Slider(
                        32,
                        400,
                        value=TOKENS_CONTINUACION,
                        step=8,
                        label="Tokens de la continuación",
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
        with gr.Tab("A cuatro manos"):
            gr.Markdown(
                "Escribí un párrafo y el escritor sigue la historia; después te toca a vos. "
                "El escritor ve las últimas ~180 palabras."
            )
            gr.ChatInterface(
                escribir_a_cuatro_manos,
                additional_inputs=[
                    gr.Radio(list(MODELOS), value=list(MODELOS)[0], label="Modelo"),
                    gr.Slider(
                        0.0, 1.5, value=TEMPERATURA_ESCRITURA, step=0.05, label="Temperatura"
                    ),
                    gr.Slider(32, 400, value=160, step=8, label="Tokens por turno"),
                ],
                additional_inputs_accordion="Opciones",
            )
    return demo


def main():
    ap = argparse.ArgumentParser(description="App de Gradio para probar el escritor.")
    ap.add_argument("--puerto", type=int, default=7860)
    ap.add_argument("--no-abrir", action="store_true", help="no abrir el navegador")
    args = ap.parse_args()
    if aviso := falta(catalogo.FINAL):
        print(f"{aviso}\n\nMientras tanto, la app arranca sin él.\n", file=sys.stderr)
    try:
        construir().launch(server_port=args.puerto, inbrowser=not args.no_abrir)
    except OSError:
        sys.exit(
            f"El puerto {args.puerto} está ocupado: seguramente la app ya está abierta en "
            f"http://127.0.0.1:{args.puerto}. Cerrala o usá --puerto {args.puerto + 1}."
        )


if __name__ == "__main__":
    main()
