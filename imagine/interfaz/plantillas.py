"""Las plantillas desde la terminal: listarlas, verlas y elegirlas con --plantilla.

uv run imagine plantillas            # lista las 12
uv run imagine plantillas soñar      # muestra el texto completo de una
"""

import argparse

from imagine.dominio.plantillas import POR_DEFECTO, TODAS, Plantilla, buscar


def argumento(nombre: str) -> Plantilla:
    """Para usar como type= de argparse: el nombre, con o sin tildes."""
    try:
        return buscar(nombre)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def agregar_opcion(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--plantilla",
        type=argumento,
        default=POR_DEFECTO,
        help=f"cómo se le presenta la escena (por defecto, {POR_DEFECTO.nombre}); "
        "ver: imagine plantillas",
    )


def main():
    ap = argparse.ArgumentParser(description="Las plantillas para presentarle la escena.")
    ap.add_argument("nombre", nargs="?", type=argumento, help="muestra su texto completo")
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
