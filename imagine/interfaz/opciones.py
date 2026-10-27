"""Las opciones del escritor en la terminal, comunes a generar y cuatro-manos."""

import argparse

from imagine.dominio.escritura import TEMPERATURA_ESCRITURA
from imagine.dominio.plantillas import POR_DEFECTO, Plantilla, buscar
from imagine.persistencia import catalogo

from .ayudas import AYUDAS


def plantilla(nombre: str) -> Plantilla:
    """Para usar como type= de argparse: el nombre, con o sin tildes."""
    try:
        return buscar(nombre)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def agregar_opciones_del_escritor(ap: argparse.ArgumentParser, largo: int) -> None:
    ap.add_argument(
        "--modelo",
        default=catalogo.FINAL,
        help="el entrenado por defecto; Qwen/Qwen3.5-0.8B-Base para comparar, u otra "
        "carpeta de runs/",
    )
    ap.add_argument(
        "--plantilla",
        type=plantilla,
        default=POR_DEFECTO,
        help=f"{AYUDAS['plantilla']} Por defecto, {POR_DEFECTO.nombre}; la lista: "
        "imagine plantillas",
    )
    ap.add_argument(
        "--temperatura", type=float, default=TEMPERATURA_ESCRITURA, help=AYUDAS["temperatura"]
    )
    ap.add_argument("--max-tokens", type=int, default=largo, help=AYUDAS["largo"])
