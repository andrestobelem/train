import math

from imagine.dominio.recompensa import recompensa


def test_sin_formato_valido_vale_cero():
    assert recompensa(None) == 0.0


def test_valida_nunca_vale_menos_que_invalida_y_crece_con_delta():
    assert recompensa(-1.0) > recompensa(None)
    # En el límite tanh satura: con Δ extremos empata con una inválida.
    assert recompensa(-10.0) >= recompensa(None)
    assert recompensa(0.0) == 1.0
    assert recompensa(0.1) > recompensa(0.0) > recompensa(-0.1)
    assert math.isclose(recompensa(0.1), 1 + math.tanh(0.4))
