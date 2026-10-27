"""El comando `imagine`: una sola puerta de entrada a las formas de usar el escritor.

uv run imagine app             # app de Gradio en el navegador
uv run imagine cuatro-manos    # historia a cuatro manos en la terminal
uv run imagine generar "…"     # imagina y continúa un fragmento
uv run imagine plantillas      # las plantillas para presentarle la escena
"""

import importlib
import os
import sys

# Subcomando → (módulo con main(), qué hace). Se importa sólo el elegido: Gradio tarda.
SUBCOMANDOS = {
    "app": ("imagine.interfaz.app", "abre la app en el navegador (fragmento y a cuatro manos)"),
    "cuatro-manos": (
        "imagine.interfaz.cuatro_manos",
        "escribe una historia por turnos en la terminal",
    ),
    "generar": ("imagine.interfaz.generar", "imagina y continúa un fragmento"),
    "plantillas": (
        "imagine.interfaz.plantillas",
        "lista las plantillas para presentarle la escena",
    ),
}


def ayuda() -> str:
    filas = "\n".join(f"  {nombre:<14}{texto}" for nombre, (_, texto) in SUBCOMANDOS.items())
    return (
        "uso: imagine <subcomando> [opciones]\n\n"
        f"{filas}\n\n"
        "Cada subcomando acepta --help para ver sus opciones."
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(ayuda())
        return
    nombre = sys.argv[1]
    if nombre not in SUBCOMANDOS:
        sys.exit(f"subcomando desconocido: {nombre}\n\n{ayuda()}")
    modulo, _ = SUBCOMANDOS[nombre]
    # El subcomando ve sus propias opciones, como si se lo hubiera llamado solo.
    sys.argv = [f"imagine {nombre}", *sys.argv[2:]]
    # Las barras de descarga de Hugging Face ensucian la terminal en cada carga.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        importlib.import_module(modulo).main()
    except KeyboardInterrupt:
        print()
        sys.exit(130)


if __name__ == "__main__":
    main()
