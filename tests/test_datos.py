import json

from imagine.datos import archivos, corpus, gutenberg, repositorio
from imagine.dominio.corpus import Ejemplo, EjemploSupervisado
from imagine.dominio.escritura import Imaginacion


def test_jsonl_ida_y_vuelta(tmp_path):
    ruta = tmp_path / "x.jsonl"
    filas = [{"texto": "año"}, {"texto": "señor"}]
    archivos.escribir_jsonl(ruta, filas)
    assert archivos.leer_jsonl(ruta) == filas
    assert archivos.leer_jsonl(ruta, 1) == filas[:1]


def test_rutas_relativas_a_la_raiz_del_proyecto():
    assert archivos.ruta_del_proyecto("data/x") == archivos.RAIZ / "data/x"
    assert (archivos.RAIZ / "pyproject.toml").exists()


def test_gutenberg_parrafos_sin_encabezado_ni_italicas():
    texto = (
        "Encabezado\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\n"
        "Primera línea\npartida _en dos_.\n\nSegundo párrafo.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***\nLicencia"
    )
    assert gutenberg.parrafos(texto) == ["Primera línea partida en dos.", "Segundo párrafo."]


def test_recortar_fin_termina_en_una_oracion_si_queda_en_la_segunda_mitad():
    texto = "Uno dos tres cuatro cinco seis. Siete ocho nueve diez"
    assert corpus.recortar_fin(texto, 8) == "Uno dos tres cuatro cinco seis."


def test_recortar_fin_corta_por_palabras_si_la_oracion_termina_muy_pronto():
    texto = "Uno dos. Tres cuatro cinco seis siete ocho nueve diez"
    assert corpus.recortar_fin(texto, 8) == "Uno dos. Tres cuatro cinco seis siete ocho"


def test_ejemplos_no_mezclan_fragmento_y_referencia():
    parrafos = [" ".join(f"p{i}w{j}" for j in range(40)) for i in range(10)]
    ejemplos = corpus.ejemplos(parrafos, "Libro")
    assert ejemplos
    for ej in ejemplos:
        assert isinstance(ej, Ejemplo) and ej.libro == "Libro"
        ultima_del_fragmento = ej.fragmento.split()[-1]
        primera_de_la_referencia = ej.referencia.split()[0]
        assert ultima_del_fragmento.split("w")[0] != primera_de_la_referencia.split("w")[0]


def test_repositorio_conserva_el_formato_en_disco(tmp_path):
    ruta = tmp_path / "ejemplos.jsonl"
    ejemplo = Ejemplo("Llovía.", "Y entró.", "Niebla")
    repositorio.escribir_ejemplos(ruta, [ejemplo])
    assert json.loads(ruta.read_text()) == {
        "libro": "Niebla",
        "contexto": "Llovía.",
        "referencia": "Y entró.",
    }
    assert repositorio.leer_ejemplos(ruta) == [ejemplo]


def test_repositorio_de_ejemplos_supervisados_ida_y_vuelta(tmp_path):
    ruta = tmp_path / "sft.jsonl"
    supervisado = EjemploSupervisado(Ejemplo("F.", "R.", "L"), Imaginacion(" I."), 0.02)
    with open(ruta, "w", encoding="utf-8") as f:
        repositorio.agregar_supervisado(f, supervisado)
    fila = json.loads(ruta.read_text())
    assert fila["imaginado"] == " I." and fila["delta"] == 0.02
    assert repositorio.leer_supervisados(ruta) == [supervisado]
