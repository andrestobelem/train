import argparse

import pytest

from imagine.dominio.plantillas import POR_DEFECTO, SONAR
from imagine.interfaz.ayudas import AYUDAS, js_que_las_muestra
from imagine.interfaz.opciones import agregar_opciones_del_escritor
from imagine.persistencia import catalogo


def opciones(*args: str, largo: int = 123) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    agregar_opciones_del_escritor(ap, largo=largo)
    return ap.parse_args(args)


def test_por_defecto_usa_el_modelo_final_y_la_plantilla_con_la_que_se_entreno():
    args = opciones()
    assert args.modelo == catalogo.FINAL
    assert args.plantilla is POR_DEFECTO
    assert args.max_tokens == 123


def test_la_plantilla_se_elige_por_nombre_con_o_sin_tildes():
    assert opciones("--plantilla", "sonar").plantilla is SONAR


def test_una_plantilla_que_no_existe_es_un_error_de_uso():
    with pytest.raises(SystemExit):
        opciones("--plantilla", "nada")


def test_la_ayuda_de_la_terminal_es_la_misma_que_la_de_la_app():
    ap = argparse.ArgumentParser()
    agregar_opciones_del_escritor(ap, largo=1)
    ayudas = {a.dest: a.help for a in ap._actions}
    assert ayudas["temperatura"] == AYUDAS["temperatura"]
    assert ayudas["max_tokens"] == AYUDAS["largo"]


def test_el_js_pone_cada_ayuda_en_su_elemento():
    js = js_que_las_muestra({"temperatura-cm": "temperatura"})
    assert '"temperatura-cm"' in js and AYUDAS["temperatura"] in js
