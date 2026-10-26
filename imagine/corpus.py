"""Descarga novelas en español de Project Gutenberg y arma ejemplos (contexto, continuación).

Cada ejemplo es un fragmento de la novela (contexto) y el texto que el autor escribió
inmediatamente después (continuación de referencia). La continuación nunca se le muestra
al modelo: sólo se usa para calcular la recompensa.
"""

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

import truststore

# Usa los certificados del sistema (necesario detrás de proxies corporativos).
truststore.inject_into_ssl()

LIBROS = {
    "train": {
        2000: "Don Quijote — Cervantes",
        17013: "Fortunata y Jacinta — Pérez Galdós",
    },
    "test": {
        49836: "Niebla — Unamuno",
    },
}

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def descargar(gid: int, destino: Path) -> str:
    ruta = destino / f"{gid}.txt"
    if not ruta.exists():
        with urllib.request.urlopen(URL.format(id=gid)) as r:
            ruta.write_bytes(r.read())
    return ruta.read_text(encoding="utf-8")


def limpiar(texto: str) -> list[str]:
    """Saca el encabezado/pie de Gutenberg y devuelve párrafos en una sola línea."""
    inicio = re.search(r"\*\*\* ?START OF.*?\*\*\*", texto)
    fin = re.search(r"\*\*\* ?END OF", texto)
    texto = texto[inicio.end() if inicio else 0 : fin.start() if fin else None]
    texto = texto.replace("\r\n", "\n")
    parrafos = []
    for bloque in re.split(r"\n\s*\n", texto):
        p = " ".join(l.strip() for l in bloque.splitlines()).strip()
        p = re.sub(r"[_*]", "", p)  # marcas de itálica de Gutenberg
        p = re.sub(r"\s+", " ", p)
        if p:
            parrafos.append(p)
    return parrafos


def recortar_inicio(texto: str, max_palabras: int) -> str:
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto
    return "… " + " ".join(palabras[-max_palabras:])


def recortar_fin(texto: str, max_palabras: int) -> str:
    """Corta al final de una oración si puede, si no por palabras."""
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto
    corto = " ".join(palabras[:max_palabras])
    corte = max(corto.rfind(". "), corto.rfind("? "), corto.rfind("! "))
    return corto[: corte + 1] if corte > len(corto) // 2 else corto


def ejemplos(parrafos, libro, ctx_palabras=180, ref_min=30, ref_max=80):
    """Ventanas deslizantes: contexto = párrafos previos, referencia = lo que sigue."""
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
        ctx = "\n\n".join(parrafos[max(0, i - 6) : i])
        if len(ctx.split()) < ctx_palabras // 2:
            continue
        out.append(
            {
                "libro": libro,
                "contexto": recortar_inicio(ctx, ctx_palabras),
                "referencia": recortar_fin(ref, ref_max),
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="data")
    ap.add_argument("--max-test", type=int, default=200)
    ap.add_argument("--semilla", type=int, default=0)
    args = ap.parse_args()

    salida = Path(args.salida)
    crudo = salida / "gutenberg"
    crudo.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(args.semilla)

    for split, libros in LIBROS.items():
        filas = []
        for gid, nombre in libros.items():
            filas += ejemplos(limpiar(descargar(gid, crudo)), nombre)
        rnd.shuffle(filas)
        if split == "test":
            filas = filas[: args.max_test]
        with open(salida / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for fila in filas:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        print(f"{split}: {len(filas)} ejemplos")


if __name__ == "__main__":
    main()
