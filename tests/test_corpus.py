import pytest

from imagine.datos.corpus import CORPUS, GUTENBERG_ID
from imagine.dominio.corpus import Corpus, Libro


def test_el_libro_de_prueba_no_puede_ser_de_un_autor_de_entrenamiento():
    with pytest.raises(ValueError, match="Pérez Galdós"):
        Corpus(
            entrenamiento=(Libro("Fortunata y Jacinta", "Pérez Galdós"),),
            prueba=(Libro("Doña Perfecta", "Pérez Galdós"),),
        )


def test_el_nombre_del_libro_es_titulo_y_autor():
    assert Libro("Niebla", "Unamuno").nombre == "Niebla — Unamuno"


def test_cada_libro_del_corpus_tiene_su_fuente():
    for libro in CORPUS.entrenamiento + CORPUS.prueba:
        assert libro.titulo in GUTENBERG_ID
