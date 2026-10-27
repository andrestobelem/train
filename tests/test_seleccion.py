from imagine.dominio.corpus import Ejemplo, EjemploSupervisado
from imagine.dominio.escritura import CIERRA, FIN_TURNO, Imaginacion
from imagine.dominio.seleccion import seleccionar

A, B, C = Imaginacion("a"), Imaginacion("b"), Imaginacion("c")


def test_elige_la_de_mayor_ganancia_entre_las_que_ayudan():
    assert seleccionar([A, B, C], [0.01, None, 0.05]) == (C, 0.05)


def test_descarta_el_fragmento_si_ninguna_ayuda():
    assert seleccionar([A, B, C], [None, -0.1, 0.0]) is None


def test_ejemplo_supervisado_enseña_imaginacion_y_referencia():
    ejemplo = Ejemplo(fragmento="Llovía.", referencia="Y entró.", libro="Niebla")
    supervisado = EjemploSupervisado(ejemplo, Imaginacion(" Augusto duda."), 0.03)
    assert supervisado.objetivo == " Augusto duda." + CIERRA + "\nY entró." + FIN_TURNO
