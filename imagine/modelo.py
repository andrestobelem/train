"""Utilidades de modelo: plantilla, muestreo en grupo y log-probabilidades por token."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODELO_BASE = "Qwen/Qwen3.5-0.8B-Base"
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


# MLX ata sus streams al hilo que los crea. Las apps con varios hilos (Inspect,
# Gradio) tienen que cargar, generar y puntuar siempre en el mismo hilo, que además
# serializa el uso de la GPU.
_HILO_MLX = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def en_hilo_mlx(fn, *args, **kwargs):
    """Corre fn en el hilo de MLX y espera el resultado (desde código sincrónico)."""
    return _HILO_MLX.submit(fn, *args, **kwargs).result()


async def en_hilo_mlx_async(fn, *args):
    """Corre fn en el hilo de MLX sin bloquear el event loop."""
    return await asyncio.get_running_loop().run_in_executor(_HILO_MLX, fn, *args)


def prompt(contexto: str) -> str:
    return PLANTILLA.format(contexto=contexto)


def cargar(adaptadores: str | None = None, base: str = MODELO_BASE):
    return load(base, adapter_path=adaptadores)


def cabeza(model, h):
    """Proyecta estados ocultos a logits (el modelo puede atar embeddings y lm_head)."""
    lm = model.language_model
    if lm.args.tie_word_embeddings:
        return lm.model.embed_tokens.as_linear(h)
    return lm.lm_head(h)


def logprobs(model, ids: mx.array, posiciones: mx.array) -> mx.array:
    """log p(ids[b, t] | ids[b, :t]) para cada t en posiciones[b] (forma B×L).

    Sólo se calculan logits en las posiciones pedidas: con un vocabulario de ~250k
    tokens, materializar logits para toda la secuencia no entra en memoria.
    """
    h = model.language_model.model(ids)
    h = mx.take_along_axis(h, (posiciones - 1)[..., None], axis=1)
    logits = cabeza(model, h).astype(mx.float32)
    objetivo = mx.take_along_axis(ids, posiciones, axis=1)
    elegido = mx.take_along_axis(logits, objetivo[..., None], axis=-1).squeeze(-1)
    return elegido - mx.logsumexp(logits, axis=-1)


def prellenar(model, prompt_ids: list[int], n: int):
    """Cache con el prompt (menos su último token) procesado, sin gradiente.

    Se usa al entrenar: la atención lineal de Qwen3.5 en modo diferenciable guarda
    un estado de 128×128 por cabeza, token y capa, así que propagar gradientes por
    un prompt de ~500 tokens no entra en memoria. Se retropropaga sólo por lo
    imaginado; el estado del prompt se trata como constante.
    """
    modo_train = model.training
    model.eval()  # kernel Metal: más rápido y sin guardar activaciones
    cache = make_prompt_cache(model)
    model(mx.array([prompt_ids[:-1]] * n), cache=cache)
    mx.eval([c.state for c in cache])
    model.train(modo_train)
    return cache


def logprobs_completacion(model, cache, ultimo: int, comps: mx.array) -> mx.array:
    """log p de cada token de comps (B×L) continuando desde un cache prellenado."""
    entrada = mx.concatenate([mx.full((comps.shape[0], 1), ultimo, dtype=comps.dtype), comps[:, :-1]], axis=1)
    h = model.language_model.model(entrada, cache)
    logits = cabeza(model, h).astype(mx.float32)
    elegido = mx.take_along_axis(logits, comps[..., None], axis=-1).squeeze(-1)
    return elegido - mx.logsumexp(logits, axis=-1)


def rellenar(secuencias: list[list[int]], pad: int) -> mx.array:
    largo = max(len(s) for s in secuencias)
    return mx.array([s + [pad] * (largo - len(s)) for s in secuencias])


def muestrear(
    model,
    tok,
    prompt_ids: list[int],
    n: int,
    max_tokens: int,
    temperatura: float,
    paradas: tuple[str, ...] = (CIERRA,),
):
    """Genera n completaciones del mismo prompt en paralelo.

    Como todas comparten el prompt no hace falta padding: el prefill se hace una vez
    con el prompt repetido y luego se decodifica token a token. Cada completación se
    corta al aparecer una de las paradas (por defecto </imagine>) o al emitir EOS.
    """
    cache = make_prompt_cache(model)
    logits = model(mx.array([prompt_ids] * n), cache=cache)[:, -1]
    salidas = [[] for _ in range(n)]
    vivas = [True] * n
    for _ in range(max_tokens):
        if temperatura == 0:
            sig = mx.argmax(logits, axis=-1)
        else:
            sig = mx.random.categorical(logits * (1 / temperatura))
        sig_l = sig.tolist()
        for i, t in enumerate(sig_l):
            if not vivas[i]:
                continue
            salidas[i].append(t)
            cola = tok.decode(salidas[i][-8:])
            if t == tok.eos_token_id or any(p in cola for p in paradas):
                vivas[i] = False
        if not any(vivas):
            break
        logits = model(sig[:, None], cache=cache)[:, -1]
    return salidas


def texto_imaginado(tok, ids: list[int]) -> tuple[str, bool]:
    """Devuelve (texto dentro de <imagine>, si cerró la etiqueta)."""
    texto = tok.decode(ids)
    if CIERRA in texto:
        return texto.split(CIERRA)[0], True
    return texto, False
