import pytest

from imagine.dominio import escritura as e
from imagine.dominio import plantillas as p


def test_son_doce_con_nombres_distintos():
    assert len(p.TODAS) == 12
    assert len({x.nombre for x in p.TODAS}) == 12


@pytest.mark.parametrize("plantilla", p.TODAS, ids=lambda x: x.nombre)
def test_cada_plantilla_presenta_el_fragmento_y_termina_abriendo_imagine(plantilla):
    texto = plantilla.prompt("FRAGMENTO")
    assert texto.endswith(e.ABRE)
    assert texto.count("FRAGMENTO") == 1
    assert "{" not in texto
    # El ejemplo enseña el formato: imagina, cierra y continúa.
    assert e.CIERRA in texto


@pytest.mark.parametrize("plantilla", p.TODAS, ids=lambda x: x.nombre)
def test_el_turno_siguiente_aparece_antes_del_fragmento(plantilla):
    """El ejemplo muestra cómo empieza el turno siguiente: así sabe dónde terminar."""
    antes, _ = plantilla.texto.split(p.FRAGMENTO)
    assert plantilla.siguiente in antes


@pytest.mark.parametrize("plantilla", p.TODAS, ids=lambda x: x.nombre)
def test_la_continuacion_se_corta_en_el_turno_siguiente(plantilla):
    texto = f"  Sigue la novela.\n{plantilla.siguiente} otro fragmento"
    assert e.cortar_continuacion(texto, plantilla) == "Sigue la novela."


def test_por_defecto_es_la_de_conversacion_con_la_que_se_entreno():
    assert p.POR_DEFECTO is p.CONVERSACION
    assert e.prompt("F.") == p.CONVERSACION.prompt("F.")
    assert e.FIN_TURNO == "\n\nLector:"


def test_buscar_ignora_tildes_y_mayusculas():
    assert p.buscar("sonar") is p.SONAR
    assert p.buscar("Folletin") is p.FOLLETIN
    with pytest.raises(ValueError, match="no hay una plantilla"):
        p.buscar("nada")
