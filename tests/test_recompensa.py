import math

from imagine.dominio.recompensa import ayuda, recompensa


def test_sin_formato_valido_vale_cero():
    assert recompensa(None) == 0.0


def test_valida_nunca_vale_menos_que_invalida_y_crece_con_la_ganancia():
    assert recompensa(-1.0) > recompensa(None)
    # En el límite tanh satura: con ganancias extremas empata con una inválida.
    assert recompensa(-10.0) >= recompensa(None)
    assert recompensa(0.0) == 1.0
    assert recompensa(0.1) > recompensa(0.0) > recompensa(-0.1)
    assert math.isclose(recompensa(0.1), 1 + math.tanh(0.4))


def test_ayuda_solo_con_ganancia_positiva():
    assert ayuda(0.01)
    assert not ayuda(0.0)
    assert not ayuda(-0.2)
    assert not ayuda(None)
