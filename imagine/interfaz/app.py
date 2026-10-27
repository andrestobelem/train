"""App local de Gradio para probar los modelos.

- Fragmento: el escritor imagina y después continúa un fragmento.
- A cuatro manos: el lector y el escritor se turnan para escribir una historia.

Cada control explica qué hace al pasarle el mouse por encima (ver ayudas.py).

uv run imagine app      # abre http://127.0.0.1:7860 en el navegador
"""

import argparse
import sys

import gradio as gr

from imagine.datos.archivos import TEST
from imagine.datos.repositorio import ejemplos_de_prueba
from imagine.dominio.escritura import (
    ABRE,
    PALABRAS_DEL_FRAGMENTO,
    TEMPERATURA_ESCRITURA,
    TOKENS_CONTINUACION,
    TOKENS_POR_TURNO,
)
from imagine.dominio.historia import Historia, Rol, Turno
from imagine.dominio.plantillas import POR_DEFECTO, TODAS, buscar
from imagine.ejecucion.escritor import imaginar_y_continuar, turno_del_escritor
from imagine.ejecucion.hilo import en_hilo_mlx
from imagine.ejecucion.modelos import MODELO_BASE, cargar_por_nombre
from imagine.persistencia import catalogo

from .ayudas import js_que_las_muestra
from .modelo import disponible, falta

TODOS = {
    "Entrenado (final)": catalogo.FINAL,
    "Sólo RL (paso 250)": catalogo.RL,
    "Sin entrenar (base)": MODELO_BASE,
}
# Sólo los que están en esta máquina: el de RL, por ejemplo, no se publica.
MODELOS = {etiqueta: nombre for etiqueta, nombre in TODOS.items() if disponible(nombre)}
PLANTILLAS = [(f"{p.nombre}: {p.descripcion}", p.nombre) for p in TODAS]

# Cada pestaña tiene sus propios controles; los de a cuatro manos llevan el sufijo -cm.
AYUDAS_POR_ELEMENTO = {
    clave: clave
    for clave in [
        "fragmento",
        "modelo",
        "plantilla",
        "imaginar",
        "temperatura",
        "largo",
        "continuar",
        "imaginacion",
        "continuacion",
    ]
} | {f"{clave}-cm": clave for clave in ["modelo", "plantilla", "temperatura", "largo"]}


def responder(
    fragmento: str, etiqueta: str, plantilla: str, imaginar: bool, temperatura: float, largo: int
):
    if not fragmento.strip():
        return "", ""

    def correr():
        model, tok = cargar_por_nombre(MODELOS[etiqueta])
        imaginacion, continuacion = imaginar_y_continuar(
            model,
            tok,
            fragmento,
            temperatura=temperatura,
            max_tokens=largo,
            con_imaginacion=imaginar,
            plantilla=buscar(plantilla),
        )
        return imaginacion.texto.strip(), continuacion

    # Gradio atiende en sus propios hilos; MLX tiene que correr siempre en el mismo.
    return en_hilo_mlx(correr)


# Título de los mensajes del chat que muestran la imaginación: no son parte de la historia.
TITULO_IMAGINACION = ABRE


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


def escribir_a_cuatro_manos(mensaje, historial, etiqueta, plantilla, temperatura, largo):
    historia = historia_desde_chat(historial, mensaje)

    def correr():
        model, tok = cargar_por_nombre(MODELOS[etiqueta])
        return turno_del_escritor(
            model,
            tok,
            historia,
            temperatura=temperatura,
            max_tokens=largo,
            plantilla=buscar(plantilla),
        )

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


def control_modelo(elem_id: str) -> gr.Radio:
    return gr.Radio(list(MODELOS), value=list(MODELOS)[0], label="Modelo", elem_id=elem_id)


def control_plantilla(elem_id: str) -> gr.Dropdown:
    return gr.Dropdown(PLANTILLAS, value=POR_DEFECTO.nombre, label="Plantilla", elem_id=elem_id)


def control_temperatura(elem_id: str) -> gr.Slider:
    return gr.Slider(
        0.0, 1.5, value=TEMPERATURA_ESCRITURA, step=0.05, label="Temperatura", elem_id=elem_id
    )


def control_largo(elem_id: str, valor: int, etiqueta: str) -> gr.Slider:
    return gr.Slider(32, 400, value=valor, step=8, label=f"{etiqueta} (tokens)", elem_id=elem_id)


def construir() -> gr.Blocks:
    with gr.Blocks(title="imagine") as demo:
        gr.Markdown(
            "# imagine\n"
            "Un modelo chico que **imagina** la escena antes de continuar una novela en "
            "español. No responde preguntas ni sigue instrucciones: continúa el texto que "
            "le des.\n\nPasá el mouse sobre un control para ver qué hace."
        )
        with gr.Tab("Fragmento"):
            with gr.Row():
                with gr.Column():
                    fragmento = gr.Textbox(
                        label="Fragmento",
                        lines=10,
                        placeholder="Pegá unas líneas de una novela y apretá Continuar.",
                        elem_id="fragmento",
                    )
                    modelo = control_modelo("modelo")
                    plantilla = control_plantilla("plantilla")
                    imaginar = gr.Checkbox(
                        value=True, label="Imaginar antes de escribir", elem_id="imaginar"
                    )
                    temperatura = control_temperatura("temperatura")
                    largo = control_largo("largo", TOKENS_CONTINUACION, "Largo de la continuación")
                    boton = gr.Button("Continuar", variant="primary", elem_id="continuar")
                with gr.Column():
                    imaginacion = gr.Textbox(label=ABRE, lines=5, elem_id="imaginacion")
                    continuacion = gr.Textbox(
                        label="Continuación", lines=12, elem_id="continuacion"
                    )
            if ejemplos():
                gr.Examples(ejemplos(), inputs=[fragmento], label="Fragmentos de Niebla")
            boton.click(
                responder,
                inputs=[fragmento, modelo, plantilla, imaginar, temperatura, largo],
                outputs=[imaginacion, continuacion],
            )
        with gr.Tab("A cuatro manos"):
            gr.Markdown(
                "Escribí un párrafo y el escritor sigue la historia; después te toca a vos. "
                f"El escritor ve las últimas ~{PALABRAS_DEL_FRAGMENTO} palabras."
            )
            gr.ChatInterface(
                escribir_a_cuatro_manos,
                additional_inputs=[
                    control_modelo("modelo-cm"),
                    control_plantilla("plantilla-cm"),
                    control_temperatura("temperatura-cm"),
                    control_largo("largo-cm", TOKENS_POR_TURNO, "Largo de cada turno"),
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
        construir().launch(
            server_port=args.puerto,
            inbrowser=not args.no_abrir,
            js=js_que_las_muestra(AYUDAS_POR_ELEMENTO),
        )
    except OSError:
        sys.exit(
            f"El puerto {args.puerto} está ocupado: seguramente la app ya está abierta en "
            f"http://127.0.0.1:{args.puerto}. Cerrala o usá --puerto {args.puerto + 1}."
        )


if __name__ == "__main__":
    main()
