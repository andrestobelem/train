"""Integración con Project Gutenberg: descarga y limpieza de los textos."""

import re
import urllib.request
from pathlib import Path

import truststore

# Usa los certificados del sistema (necesario detrás de proxies corporativos).
truststore.inject_into_ssl()

URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"


def descargar(gid: int, destino: Path) -> str:
    """El texto del libro, descargado una sola vez y guardado en destino."""
    ruta = destino / f"{gid}.txt"
    if not ruta.exists():
        with urllib.request.urlopen(URL.format(id=gid)) as r:
            ruta.write_bytes(r.read())
    return ruta.read_text(encoding="utf-8")


def parrafos(texto: str) -> list[str]:
    """Saca el encabezado/pie de Gutenberg y devuelve párrafos en una sola línea."""
    inicio = re.search(r"\*\*\* ?START OF.*?\*\*\*", texto)
    fin = re.search(r"\*\*\* ?END OF", texto)
    texto = texto[inicio.end() if inicio else 0 : fin.start() if fin else None]
    texto = texto.replace("\r\n", "\n")
    salida = []
    for bloque in re.split(r"\n\s*\n", texto):
        p = " ".join(linea.strip() for linea in bloque.splitlines()).strip()
        p = re.sub(r"[_*]", "", p)  # marcas de itálica de Gutenberg
        p = re.sub(r"\s+", " ", p)
        if p:
            salida.append(p)
    return salida
