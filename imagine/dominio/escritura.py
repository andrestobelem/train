"""La escena de escritura: imaginación, prompts y dónde termina cada turno.

Un lector comparte un fragmento; el escritor imagina entre <imagine> </imagine> y
después continúa la novela. Los términos están definidos en GLOSSARY.md.
"""

from dataclasses import dataclass

from .etiquetas import ABRE, CIERRA
from .plantillas import POR_DEFECTO, Plantilla

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
# En una historia a cuatro manos, cada turno es más corto.
TOKENS_POR_TURNO = 160

# Con la plantilla por defecto: el turno siguiente marca el final de la continuación.
FIN_TURNO = POR_DEFECTO.fin_turno
FINES_CONTINUACION = POR_DEFECTO.fines


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


def prompt(fragmento: str, plantilla: Plantilla = POR_DEFECTO) -> str:
    """Lo que recibe el escritor: termina abriendo <imagine>."""
    return plantilla.prompt(fragmento)


def prompt_continuacion(
    fragmento: str, imaginacion: Imaginacion, plantilla: Plantilla = POR_DEFECTO
) -> str:
    """El prompt con la imaginación ya cerrada: el escritor sigue con la continuación."""
    return prompt(fragmento, plantilla) + imaginacion.texto + CIERRA + "\n"


def cortar_continuacion(texto: str, plantilla: Plantilla = POR_DEFECTO) -> str:
    """La continuación termina donde empieza otro turno o el escritor vuelve a las etiquetas."""
    for fin in plantilla.fines:
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
