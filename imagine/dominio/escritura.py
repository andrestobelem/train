"""La escena de escritura: plantilla, etiquetas, imaginación y dónde termina cada turno.

Un lector comparte un fragmento; el escritor imagina entre <imagine> </imagine> y
después continúa la novela. Los términos están definidos en GLOSSARY.md.
"""

from dataclasses import dataclass

ABRE, CIERRA = "<imagine>", "</imagine>"

# Adaptación de la plantilla de DeepSeek-R1-Zero (tabla 1 del paper): en lugar de
# razonar dentro de <think> antes de responder, el escritor imagina dentro de
# <imagine> antes de continuar la novela. A diferencia de R1-Zero se incluye un
# ejemplo (one-shot): un modelo de 0.8B casi nunca cierra la etiqueta por sí solo.
PLANTILLA = (
    "Un lector y un escritor conversan. El lector comparte un fragmento de una novela "
    "y el escritor la continúa. Antes de escribir, el escritor imagina brevemente lo que "
    "viene: qué sienten los personajes, qué van a decir o hacer y cómo sigue la escena. "
    f"Lo imaginado va entre {ABRE} {CIERRA} y después viene la continuación.\n\n"
    "Lector: Fragmento:\n"
    "La lluvia no paraba. Elena dejó la carta sobre la mesa sin abrirla y miró a su "
    "hermano, que fingía leer el periódico junto a la ventana.\n"
    f"Escritor: {ABRE} Elena sospecha que la carta es del padre que los abandonó. El "
    "hermano lo sabe y la evita. Ella lo enfrenta y él termina confesando que le escribía "
    f"a escondidas. {CIERRA}\n"
    "—¿No pensás preguntarme de quién es? —dijo Elena.\n\n"
    "Lector: Fragmento:\n{fragmento}\n"
    f"Escritor: {ABRE}"
)

# Cuántas palabras del texto previo entran en un fragmento (así se armaron los ejemplos).
PALABRAS_DEL_FRAGMENTO = 180

# Una imaginación válida tiene al menos estas palabras.
MIN_PALABRAS = 5

# Presupuesto de la imaginación, el mismo al entrenar, evaluar y usar el modelo.
TOKENS_IMAGINACION = 96

# Temperatura para muestrear imaginaciones al entrenar y evaluar (explorar variantes).
TEMPERATURA_MUESTREO = 0.8

# Temperatura y largo por defecto al escribir para un lector (más conservador).
TEMPERATURA_ESCRITURA = 0.7
TOKENS_CONTINUACION = 200

# El próximo turno del lector: marca el final de la continuación.
FIN_TURNO = "\n\nLector:"

# Dónde se corta una continuación: otro turno, o el modelo volviendo a las etiquetas.
FINES_CONTINUACION = ("\nLector:", ABRE, CIERRA)


@dataclass(frozen=True)
class Imaginacion:
    """Lo que el escritor escribe entre <imagine> y </imagine>."""

    texto: str
    cerro: bool = True  # si cerró la etiqueta

    @classmethod
    def desde_generacion(cls, generado: str) -> "Imaginacion":
        """Lo que el escritor generó después de <imagine>, hasta el cierre si lo hubo."""
        if CIERRA in generado:
            return cls(generado.split(CIERRA)[0], True)
        return cls(generado, False)

    @property
    def formato_valido(self) -> bool:
        """Cierra la etiqueta, no la vuelve a abrir y tiene al menos MIN_PALABRAS."""
        return self.cerro and ABRE not in self.texto and len(self.texto.split()) >= MIN_PALABRAS


SIN_IMAGINACION = Imaginacion("")


def prompt(fragmento: str) -> str:
    """Lo que recibe el escritor: termina abriendo <imagine>."""
    return PLANTILLA.format(fragmento=fragmento)


def prompt_continuacion(fragmento: str, imaginacion: Imaginacion) -> str:
    """El prompt con la imaginación ya cerrada: el escritor sigue con la continuación."""
    return prompt(fragmento) + imaginacion.texto + CIERRA + "\n"


def cortar_continuacion(texto: str) -> str:
    """La continuación termina donde empieza otro turno o el escritor vuelve a las etiquetas."""
    for fin in FINES_CONTINUACION:
        texto = texto.split(fin)[0]
    return texto.strip()


def objetivo_supervisado(imaginacion: Imaginacion, referencia: str) -> str:
    """Lo que el ajuste supervisado enseña tras el prompt: imaginar, continuar como el autor
    y ceder el turno."""
    return imaginacion.texto + CIERRA + "\n" + referencia + FIN_TURNO


def recortar_fragmento(texto: str, max_palabras: int = PALABRAS_DEL_FRAGMENTO) -> str:
    """El final del texto previo que ve el escritor, marcado con … si se recortó."""
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto
    return "… " + " ".join(palabras[-max_palabras:])
