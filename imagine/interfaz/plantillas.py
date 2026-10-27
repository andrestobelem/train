"""Las plantillas desde la terminal: listarlas y ver el texto de cada una.

uv run imagine plantillas            # lista las 12
uv run imagine plantillas soñar      # muestra el texto completo de una
"""

import argparse

from imagine.dominio.plantillas import TODAS

from .opciones import plantilla


def main():
    ap = argparse.ArgumentParser(description="Las plantillas para presentarle la escena.")
    ap.add_argument("nombre", nargs="?", type=plantilla, help="muestra su texto completo")
    args = ap.parse_args()
    if args.nombre is not None:
        print(args.nombre.prompt("{fragmento}"))
        return
    ancho = max(len(p.nombre) for p in TODAS)
    for p in TODAS:
        print(f"  {p.nombre:<{ancho}}  {p.descripcion}")
    print("\nEl modelo se entrenó sólo con «conversación»; las demás son para explorar.")
    print('Uso: imagine generar --plantilla soñar "…"')


if __name__ == "__main__":
    main()
