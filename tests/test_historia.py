import pytest

from imagine.dominio.escritura import PALABRAS_DEL_FRAGMENTO
from imagine.dominio.historia import Historia, Rol, Turno


def test_los_turnos_alternan_y_empieza_el_lector():
    historia = Historia()
    assert historia.le_toca is Rol.LECTOR
    historia = historia.agregar(Turno(Rol.LECTOR, "Llovía."))
    assert historia.le_toca is Rol.ESCRITOR
    with pytest.raises(ValueError, match="escritor"):
        historia.agregar(Turno(Rol.LECTOR, "Otra vez yo."))


def test_un_turno_vacio_pasa_sin_agregar_texto():
    historia = (
        Historia()
        .agregar(Turno(Rol.LECTOR, "Llovía."))
        .agregar(Turno(Rol.ESCRITOR, ""))
        .agregar(Turno(Rol.LECTOR, "Y seguía lloviendo."))
    )
    assert historia.texto == "Llovía.\n\nY seguía lloviendo."
    assert historia.le_toca is Rol.ESCRITOR


def test_el_escritor_ve_el_final_de_la_historia_como_fragmento():
    largo = " ".join(f"p{i}" for i in range(PALABRAS_DEL_FRAGMENTO + 20))
    historia = Historia().agregar(Turno(Rol.LECTOR, largo))
    assert historia.fragmento.startswith("… p20 ")
    assert historia.fragmento.endswith(f"p{PALABRAS_DEL_FRAGMENTO + 19}")


def test_historia_desde_chat_omite_la_imaginacion():
    from imagine.interfaz.app import TITULO_IMAGINACION, historia_desde_chat

    chat = [
        {"role": "user", "content": "Llovía."},
        {"role": "assistant", "content": "plan", "metadata": {"title": TITULO_IMAGINACION}},
        {"role": "assistant", "content": "Augusto entró."},
    ]
    historia = historia_desde_chat(chat, "Y Eugenia lo vio.")
    assert [t.rol for t in historia.turnos] == [Rol.LECTOR, Rol.ESCRITOR, Rol.LECTOR]
    assert historia.le_toca is Rol.ESCRITOR
