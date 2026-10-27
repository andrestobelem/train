"""El protocolo de escritura: plantilla, etiquetas y dónde empieza y termina cada parte.

Un lector comparte un fragmento; el escritor imagina entre <imagine> </imagine> y
después continúa la novela. Estas reglas las comparten el entrenamiento, la
evaluación y la interfaz, así que viven en un solo lugar.
"""

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
    "Lector: Fragmento:\n{contexto}\n"
    f"Escritor: {ABRE}"
)

# Cuántas palabras del texto previo ve el escritor (así se armaron los ejemplos).
PALABRAS_DE_CONTEXTO = 180

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


def prompt(fragmento: str) -> str:
    """El texto que recibe el modelo: termina abriendo <imagine>."""
    return PLANTILLA.format(contexto=fragmento)


def prompt_continuacion(fragmento: str, imaginado: str) -> str:
    """El prompt con lo imaginado ya cerrado: el modelo sigue con la continuación."""
    return prompt(fragmento) + imaginado + CIERRA + "\n"


def separar_imaginado(texto: str) -> tuple[str, bool]:
    """Lo generado tras <imagine> → (lo imaginado, si cerró la etiqueta)."""
    if CIERRA in texto:
        return texto.split(CIERRA)[0], True
    return texto, False


def formato_ok(imaginado: str, cerro: bool) -> bool:
    """La regla de formato de la recompensa: cerró, no reabrió y no está casi vacío."""
    return cerro and ABRE not in imaginado and len(imaginado.split()) >= MIN_PALABRAS


def cortar_continuacion(texto: str) -> str:
    for fin in FINES_CONTINUACION:
        texto = texto.split(fin)[0]
    return texto.strip()


def objetivo_supervisado(imaginado: str, referencia: str) -> str:
    """Lo que el SFT enseña a escribir tras el prompt: imaginar, continuar y ceder el turno."""
    return imaginado + CIERRA + "\n" + referencia + FIN_TURNO


def recortar_inicio(texto: str, max_palabras: int = PALABRAS_DE_CONTEXTO) -> str:
    """El final del texto que ve el escritor, marcado con … si se recortó."""
    palabras = texto.split()
    if len(palabras) <= max_palabras:
        return texto
    return "… " + " ".join(palabras[-max_palabras:])
