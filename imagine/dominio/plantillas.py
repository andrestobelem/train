"""Las plantillas: distintos encuadres para presentarle la escena al escritor.

Todas tienen una consigna, un ejemplo resuelto y el fragmento, y terminan abriendo
<imagine>. Cambian el encuadre (conversar, soñar, leer, poner en escena…) y con él el
tipo de imaginación que se le pide. El modelo se entrenó sólo con la de conversación;
las demás se pueden elegir para explorar, sin reentrenar.
"""

import unicodedata
from dataclasses import dataclass

from .etiquetas import ABRE, CIERRA

FRAGMENTO = "{fragmento}"


@dataclass(frozen=True)
class Plantilla:
    nombre: str
    descripcion: str
    texto: str  # con {fragmento}; termina abriendo <imagine>
    siguiente: str  # cómo empieza el turno siguiente: ahí termina la continuación

    def prompt(self, fragmento: str) -> str:
        return self.texto.replace(FRAGMENTO, fragmento)

    @property
    def fines(self) -> tuple[str, ...]:
        """Dónde se corta una continuación: otro turno, o el escritor volviendo a las etiquetas."""
        return ("\n" + self.siguiente, ABRE, CIERRA)

    @property
    def fin_turno(self) -> str:
        """Cómo cede el turno el escritor al terminar de escribir."""
        return "\n\n" + self.siguiente


# El mismo ejemplo en todas, para que lo único que cambie sea el encuadre.
EJEMPLO = (
    "La lluvia no paraba. Elena dejó la carta sobre la mesa sin abrirla y miró a su "
    "hermano, que fingía leer el periódico junto a la ventana."
)
# Fuera de la conversación, el ejemplo sigue con narración y no con diálogo.
NARRACION = (
    "Afuera pasó un carro salpicando el empedrado. Elena esperó a que el ruido se apagara, "
    "pero el hermano dio vuelta la página como si en la sala no hubiera nadie más."
)
SOSPECHA = (
    "Elena sospecha que la carta es del padre que los abandonó. El hermano lo sabe y la evita."
)

# Adaptación de la plantilla de DeepSeek-R1-Zero (tabla 1 del paper): en lugar de
# razonar dentro de <think> antes de responder, el escritor imagina dentro de
# <imagine> antes de continuar la novela. A diferencia de R1-Zero se incluye un
# ejemplo (one-shot): un modelo de 0.8B casi nunca cierra la etiqueta por sí solo.
CONVERSACION = Plantilla(
    "conversación",
    "un lector comparte un fragmento y el escritor lo continúa (con la que se entrenó)",
    "Un lector y un escritor conversan. El lector comparte un fragmento de una novela "
    "y el escritor la continúa. Antes de escribir, el escritor imagina brevemente lo que "
    "viene: qué sienten los personajes, qué van a decir o hacer y cómo sigue la escena. "
    f"Lo imaginado va entre {ABRE} {CIERRA} y después viene la continuación.\n\n"
    "Lector: Fragmento:\n"
    f"{EJEMPLO}\n"
    f"Escritor: {ABRE} Elena sospecha que la carta es del padre que los abandonó. El "
    "hermano lo sabe y la evita. Ella lo enfrenta y él termina confesando que le escribía "
    f"a escondidas. {CIERRA}\n"
    "—¿No pensás preguntarme de quién es? —dijo Elena.\n\n"
    "Lector: Fragmento:\n{fragmento}\n"
    f"Escritor: {ABRE}",
    "Lector:",
)

NOVELA = Plantilla(
    "novela",
    "pasajes de novela, sin roles ni conversación",
    "Pasajes de novelas en español. Antes de continuar cada pasaje, quien escribe imagina "
    f"brevemente lo que viene entre {ABRE} {CIERRA} y después lo continúa.\n\n"
    f"Pasaje:\n{EJEMPLO}\n{ABRE} {SOSPECHA} {CIERRA}\n{NARRACION}\n\n"
    f"Pasaje:\n{{fragmento}}\n{ABRE}",
    "Pasaje:",
)

IMAGINAR = Plantilla(
    "imaginar",
    "ver la escena con los sentidos: lugar, sonidos, objetos",
    "Ejercicio de imaginación. Antes de seguir el texto, quien escribe cierra los ojos y "
    "ve la escena: dónde están los personajes, qué se oye, qué tienen en las manos, qué "
    f"les pasa por dentro. Lo que ve lo anota entre {ABRE} {CIERRA} y después escribe.\n\n"
    f"Texto:\n{EJEMPLO}\n"
    f"{ABRE} Una sala en penumbra, la lluvia contra el vidrio. El sobre está cerrado y "
    "tiene una letra que Elena reconoce. El hermano no lee: hace rato que no pasa la "
    f"página. {CIERRA}\n{NARRACION}\n\n"
    f"Texto:\n{{fragmento}}\n{ABRE}",
    "Texto:",
)

SONAR = Plantilla(
    "soñar",
    "soñar cómo sigue antes de escribirlo",
    "Antes de escribir, la autora se duerme sobre el manuscrito y sueña cómo sigue. Al "
    f"despertar anota el sueño entre {ABRE} {CIERRA}, tal como lo recuerda, y después "
    "escribe la continuación despierta.\n\n"
    f"Manuscrito:\n{EJEMPLO}\n"
    f"{ABRE} Soñé que la carta pesaba como una piedra y que la lluvia entraba por debajo "
    f"de la puerta. El hermano tenía la cara del padre. Nadie la abría. {CIERRA}\n"
    f"{NARRACION}\n\n"
    f"Manuscrito:\n{{fragmento}}\n{ABRE}",
    "Manuscrito:",
)

LEER = Plantilla(
    "leer",
    "un lector que adivina qué va a pasar antes de dar vuelta la página",
    "Un lector lee una novela. Al final de cada página, antes de darla vuelta, anota "
    f"entre {ABRE} {CIERRA} lo que cree que va a pasar. Después sigue leyendo.\n\n"
    f"Página:\n{EJEMPLO}\n"
    f"{ABRE} Creo que Elena no va a abrir la carta todavía: va a esperar a que el hermano "
    f"diga algo, y él va a seguir haciéndose el distraído. {CIERRA}\n{NARRACION}\n\n"
    f"Página:\n{{fragmento}}\n{ABRE}",
    "Página:",
)

ESCRIBIR = Plantilla(
    "escribir",
    "un novelista decide qué tiene que lograr el párrafo siguiente",
    "Un novelista trabaja en su escritorio. Antes de cada párrafo nuevo, decide entre "
    f"{ABRE} {CIERRA} qué tiene que lograr: qué se revela, qué se calla, cómo termina. "
    "Después lo escribe.\n\n"
    f"Último párrafo:\n{EJEMPLO}\n"
    f"{ABRE} Todavía no revelar de quién es la carta. Que la tensión venga del silencio "
    f"del hermano. Cerrar con un gesto, no con una frase. {CIERRA}\n{NARRACION}\n\n"
    f"Último párrafo:\n{{fragmento}}\n{ABRE}",
    "Último párrafo:",
)

CUATRO_MANOS = Plantilla(
    "cuatro-manos",
    "dos autores que se turnan un párrafo cada uno",
    "Dos autores escriben una novela a cuatro manos: cada uno escribe un párrafo y le pasa "
    "la pluma al otro. Antes de escribir, cada autor anota brevemente entre "
    f"{ABRE} {CIERRA} cómo sigue la historia.\n\n"
    f"Autor A: {EJEMPLO}\n"
    f"Autor B: {ABRE} {SOSPECHA} {CIERRA}\n{NARRACION}\n\n"
    f"Autor A: {{fragmento}}\n"
    f"Autor B: {ABRE}",
    "Autor A:",
)

TALLER = Plantilla(
    "taller",
    "continuar a otro autor sin que se note el cambio de mano",
    "Taller de escritura. Consigna: continuar el texto de otro autor sin que se note el "
    f"cambio de mano. Antes de escribir, cada participante anota entre {ABRE} {CIERRA} "
    "qué leyó en la escena: el tono, lo que sienten los personajes y lo que está por "
    "pasar.\n\n"
    f"Texto del autor: {EJEMPLO}\n"
    f"Participante: {ABRE} Tono contenido, de silencios. {SOSPECHA} {CIERRA}\n"
    f"{NARRACION}\n\n"
    f"Texto del autor: {{fragmento}}\n"
    f"Participante: {ABRE}",
    "Texto del autor:",
)

FOLLETIN = Plantilla(
    "folletín",
    "novela por entregas: qué pasa en la próxima",
    "Novela por entregas. Cada entrega retoma el final de la anterior. Antes de "
    f"escribirla, el folletinista anota entre {ABRE} {CIERRA} qué va a pasar en la "
    "entrega.\n\n"
    f"Final de la entrega anterior:\n{EJEMPLO}\n"
    f"Entrega siguiente:\n{ABRE} {SOSPECHA} Ella lo enfrenta y él termina confesando "
    f"que le escribía a escondidas. {CIERRA}\n{NARRACION}\n\n"
    f"Final de la entrega anterior:\n{{fragmento}}\n"
    f"Entrega siguiente:\n{ABRE}",
    "Final de la entrega anterior:",
)

TEATRO = Plantilla(
    "teatro",
    "poner la escena: dónde está cada uno, gestos y silencios",
    "Una directora adapta una novela al teatro. Antes de cada escena, anota entre "
    f"{ABRE} {CIERRA} cómo la pondría: dónde está cada personaje, qué gestos hace, "
    "dónde van los silencios. Después escribe la escena como prosa de novela.\n\n"
    f"Escena:\n{EJEMPLO}\n"
    f"{ABRE} Elena de pie junto a la mesa; el hermano sentado, de espaldas, contra la "
    "ventana. Un silencio largo. Ella no toca la carta. Él da vuelta una página sin "
    f"leerla. {CIERRA}\n{NARRACION}\n\n"
    f"Escena:\n{{fragmento}}\n{ABRE}",
    "Escena:",
)

PERSONAJE = Plantilla(
    "personaje",
    "ponerse en la piel del personaje y hablar en primera persona",
    "Antes de seguir la novela, quien escribe se pone en la piel del personaje principal "
    f"de la escena y anota entre {ABRE} {CIERRA}, en primera persona, lo que siente y "
    "lo que quiere en ese momento. Después vuelve a la voz del narrador y continúa.\n\n"
    f"Fragmento:\n{EJEMPLO}\n"
    f"{ABRE} No la voy a abrir delante de él. Sé de quién es. Quiero que sea él quien lo "
    f"diga primero. {CIERRA}\n{NARRACION}\n\n"
    f"Fragmento:\n{{fragmento}}\n{ABRE}",
    "Fragmento:",
)

RECORDAR = Plantilla(
    "recordar",
    "una autora que escribe lo que vivió y primero lo recuerda",
    "Una autora escribe una novela sobre algo que vivió. Antes de cada parte, recuerda "
    f"entre {ABRE} {CIERRA} cómo fue de verdad: lo que se dijo, lo que no, cómo olía la "
    "casa. Después lo escribe como novela.\n\n"
    f"Lo que llevo escrito:\n{EJEMPLO}\n"
    f"{ABRE} Me acuerdo del olor a papel mojado y de que mi hermano tenía las manos "
    f"quietas. La carta estuvo sobre la mesa toda la tarde. {CIERRA}\n{NARRACION}\n\n"
    f"Lo que llevo escrito:\n{{fragmento}}\n{ABRE}",
    "Lo que llevo escrito:",
)

TODAS = (
    CONVERSACION,
    NOVELA,
    IMAGINAR,
    SONAR,
    LEER,
    ESCRIBIR,
    CUATRO_MANOS,
    TALLER,
    FOLLETIN,
    TEATRO,
    PERSONAJE,
    RECORDAR,
)
POR_DEFECTO = CONVERSACION


def _clave(nombre: str) -> str:
    """Sin tildes ni mayúsculas, para que «sonar» encuentre «soñar»."""
    sin_marcas = unicodedata.normalize("NFD", nombre.lower())
    return "".join(c for c in sin_marcas if not unicodedata.combining(c))


_POR_CLAVE = {_clave(p.nombre): p for p in TODAS}


def buscar(nombre: str) -> Plantilla:
    try:
        return _POR_CLAVE[_clave(nombre)]
    except KeyError:
        nombres = ", ".join(p.nombre for p in TODAS)
        raise ValueError(f"no hay una plantilla «{nombre}»; las que hay: {nombres}") from None
