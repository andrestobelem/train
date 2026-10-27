"""Arma los ejemplos (fragmento y referencia) a partir de los libros.

La referencia es lo que el autor escribió inmediatamente después del fragmento. El
escritor nunca la ve: sólo se usa para medir la ganancia y para el ajuste supervisado.

    uv run python -m imagine.datos.corpus
"""

import argparse
import random

from imagine.dominio.corpus import Ejemplo
from imagine.dominio.escritura import PALABRAS_DEL_FRAGMENTO, recortar_fragmento

from . import gutenberg
from .archivos import DATOS
from .repositorio import escribir_ejemplos

LIBROS = {
    "train": {
        2000: "Don Quijote — Cervantes",
        17013: "Fortunata y Jacinta — Pérez Galdós",
    },
    "test": {
        49836: "Niebla — Unamuno",
    },
}


def recortar_fin(texto: str, max_palabras: int) -> str:
    """Corta al final de una oración si puede, si no por palabras."""
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto
    corto = " ".join(palabras[:max_palabras])
    corte = max(corto.rfind(". "), corto.rfind("? "), corto.rfind("! "))
    return corto[: corte + 1] if corte > len(corto) // 2 else corto


def ejemplos(
    parrafos: list[str],
    libro: str,
    max_fragmento: int = PALABRAS_DEL_FRAGMENTO,
    ref_min: int = 30,
    ref_max: int = 80,
) -> list[Ejemplo]:
    """Ventanas deslizantes: el fragmento son los párrafos previos y la referencia, lo que
    sigue (al menos ref_min palabras, aunque abarque varios párrafos)."""
    out = []
    for i in range(3, len(parrafos)):
        ref = parrafos[i]
        j = i + 1
        while len(ref.split()) < ref_min and j < len(parrafos):
            ref += "\n\n" + parrafos[j]
            j += 1
        # Párrafos muy cortos suelen ser títulos de capítulo o notas.
        if len(ref.split()) < ref_min or len(parrafos[i].split()) < 8:
            continue
        previo = "\n\n".join(parrafos[max(0, i - 6) : i])
        if len(previo.split()) < max_fragmento // 2:
            continue
        out.append(
            Ejemplo(
                fragmento=recortar_fragmento(previo, max_fragmento),
                referencia=recortar_fin(ref, ref_max),
                libro=libro,
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-test", type=int, default=200)
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    crudo = DATOS / "gutenberg"
    crudo.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(args.semilla)

    for split, libros in LIBROS.items():
        filas = []
        for gid, nombre in libros.items():
            filas += ejemplos(gutenberg.parrafos(gutenberg.descargar(gid, crudo)), nombre)
        rnd.shuffle(filas)
        if split == "test":
            filas = filas[: args.max_test]
        escribir_ejemplos(DATOS / f"{split}.jsonl", filas)
        print(f"{split}: {len(filas)} ejemplos")


if __name__ == "__main__":
    main()
