"""Qué hace cada control: lo mismo en la app (al pasar el mouse) y en --help de la terminal."""

import json

from imagine.dominio.escritura import PALABRAS_DEL_FRAGMENTO

AYUDAS = {
    "fragmento": "El texto que el modelo va a continuar. Funciona mejor con prosa narrativa "
    "en español. Si es largo, el modelo ve sólo las últimas "
    f"{PALABRAS_DEL_FRAGMENTO} palabras.",
    "modelo": "Entrenado: el de este proyecto. Sin entrenar: el modelo original, para "
    "comparar. Sólo RL: la etapa intermedia del entrenamiento, si está en esta máquina.",
    "plantilla": "Cómo se le presenta la escena: como una conversación, un sueño, una "
    "lectura, una puesta en escena… Cambia qué tipo de cosas imagina. El modelo se entrenó "
    "sólo con «conversación»; las demás son para explorar.",
    "imaginar": "Apagado, el modelo continúa directamente, sin imaginar antes. Sirve para "
    "comparar qué cambia la imaginación.",
    "temperatura": "Cuánto se arriesga al elegir cada palabra. En 0 elige siempre la más "
    "probable y tiende a repetirse. Alrededor de 0,7 hay equilibrio. Por encima de 1 "
    "sorprende más, pero pierde coherencia.",
    "largo": "Largo máximo de lo que escribe, en tokens: pedazos de palabra. 100 tokens "
    "son unas 65 palabras. Puede terminar antes si cede el turno.",
    "continuar": "Imagina y después continúa el fragmento. Tarda unos segundos.",
    "imaginacion": "La nota que el modelo escribió para sí mismo antes de continuar. No "
    "forma parte del texto.",
    "continuacion": "Lo que el modelo escribió después de imaginar.",
}


def js_que_las_muestra(por_elemento: dict[str, str]) -> str:
    """JS para la app: pone cada ayuda como title de su elemento (elem_id → clave de AYUDAS).

    Gradio no tiene tooltips. Un observador las vuelve a poner en los elementos que
    aparecen después, como los de otra pestaña.
    """
    textos = {elem_id: AYUDAS[clave] for elem_id, clave in por_elemento.items()}
    return f"""() => {{
    const ayudas = {json.dumps(textos, ensure_ascii=False)};
    const poner = () => {{
        for (const [id, texto] of Object.entries(ayudas)) {{
            const el = document.getElementById(id);
            if (el && el.title !== texto) el.title = texto;
        }}
    }};
    poner();
    new MutationObserver(poner).observe(document.body, {{childList: true, subtree: true}});
}}"""
