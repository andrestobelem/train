from imagine.dominio import escritura as e
from imagine.dominio.escritura import Imaginacion


def test_prompt_termina_abriendo_imagine_y_contiene_el_fragmento():
    p = e.prompt("Llovía sobre Salamanca.")
    assert p.endswith(f"Escritor: {e.ABRE}")
    assert "Lector: Fragmento:\nLlovía sobre Salamanca.\n" in p


def test_prompt_continuacion_cierra_la_imaginacion():
    p = e.prompt_continuacion("Llovía.", Imaginacion(" Augusto duda."))
    assert p == e.prompt("Llovía.") + " Augusto duda." + e.CIERRA + "\n"


def test_imaginacion_desde_generacion():
    assert Imaginacion.desde_generacion(" Augusto duda. </imagine>\nSigue") == Imaginacion(
        " Augusto duda. ", cerro=True
    )
    assert Imaginacion.desde_generacion(" sin cerrar") == Imaginacion(" sin cerrar", cerro=False)


def test_formato_valido_exige_cierre_minimo_de_palabras_y_no_reabrir():
    largo = "una dos tres cuatro cinco"
    assert Imaginacion(largo).formato_valido
    assert not Imaginacion(largo, cerro=False).formato_valido
    assert not Imaginacion("muy corto").formato_valido
    assert not Imaginacion(largo + " " + e.ABRE).formato_valido


def test_sin_imaginacion_no_tiene_formato_valido_pero_sirve_para_continuar():
    assert not e.SIN_IMAGINACION.formato_valido
    assert e.prompt_continuacion("F.", e.SIN_IMAGINACION).endswith(e.ABRE + e.CIERRA + "\n")


def test_cortar_continuacion_en_cada_fin():
    for fin in e.FINES_CONTINUACION:
        assert e.cortar_continuacion(f"  Texto.{fin}resto") == "Texto."


def test_objetivo_supervisado_imagina_continua_y_cede_el_turno():
    objetivo = e.objetivo_supervisado(Imaginacion(" Plan."), "Lo que escribió el autor.")
    assert objetivo == " Plan." + e.CIERRA + "\nLo que escribió el autor." + e.FIN_TURNO


def test_recortar_fragmento_conserva_el_final():
    texto = " ".join(str(i) for i in range(300))
    assert e.recortar_fragmento(texto, 10) == "… " + " ".join(str(i) for i in range(290, 300))
    assert e.recortar_fragmento("corto", 10) == "corto"
