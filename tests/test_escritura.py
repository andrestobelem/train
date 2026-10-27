from imagine.dominio import escritura as e


def test_prompt_termina_abriendo_imagine_y_contiene_el_fragmento():
    p = e.prompt("Llovía sobre Salamanca.")
    assert p.endswith(f"Escritor: {e.ABRE}")
    assert "Lector: Fragmento:\nLlovía sobre Salamanca.\n" in p


def test_prompt_continuacion_cierra_lo_imaginado():
    p = e.prompt_continuacion("Llovía.", " Augusto duda.")
    assert p == e.prompt("Llovía.") + " Augusto duda." + e.CIERRA + "\n"


def test_separar_imaginado():
    assert e.separar_imaginado(" Augusto duda. </imagine>\nSigue") == (" Augusto duda. ", True)
    assert e.separar_imaginado(" sin cerrar") == (" sin cerrar", False)


def test_formato_ok_exige_cierre_minimo_de_palabras_y_no_reabrir():
    largo = "una dos tres cuatro cinco"
    assert e.formato_ok(largo, cerro=True)
    assert not e.formato_ok(largo, cerro=False)
    assert not e.formato_ok("muy corto", cerro=True)
    assert not e.formato_ok(largo + " " + e.ABRE, cerro=True)


def test_cortar_continuacion_en_cada_fin():
    for fin in e.FINES_CONTINUACION:
        assert e.cortar_continuacion(f"  Texto.{fin}resto") == "Texto."


def test_objetivo_supervisado_imagina_continua_y_cede_el_turno():
    objetivo = e.objetivo_supervisado(" Plan.", "Lo que escribió el autor.")
    assert objetivo == " Plan." + e.CIERRA + "\nLo que escribió el autor." + e.FIN_TURNO


def test_recortar_inicio_conserva_el_final():
    texto = " ".join(str(i) for i in range(300))
    recortado = e.recortar_inicio(texto, 10)
    assert recortado == "… " + " ".join(str(i) for i in range(290, 300))
    assert e.recortar_inicio("corto", 10) == "corto"
