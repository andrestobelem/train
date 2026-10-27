from imagine.dominio import prosa


def test_repeticion_y_distintos():
    assert prosa.repeticion("a b c d e") == 0.0
    assert prosa.repeticion("a b c a b c") == 0.25  # 4 trigramas, 3 distintos
    assert prosa.distintos("a b a b") == 2 / 3


def test_textos_cortos_no_rompen():
    assert prosa.repeticion("") == 0.0
    assert prosa.distintos("una") == 0.0
    assert prosa.ajenos("") == 0.0


def test_ajenos_cuenta_letras_fuera_del_alfabeto_latino():
    assert prosa.ajenos("Ñandú, pingüino y cigüeña") == 0.0
    assert prosa.ajenos("abαβ") == 0.5
