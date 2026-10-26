"""Operaciones de bajo nivel sobre el modelo de lenguaje: muestreo y log-probabilidades."""

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache


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
    generado; el estado del prompt se trata como constante.
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
    primero = mx.full((comps.shape[0], 1), ultimo, dtype=comps.dtype)
    h = model.language_model.model(mx.concatenate([primero, comps[:, :-1]], axis=1), cache)
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
    paradas: tuple[str, ...] = (),
) -> list[list[int]]:
    """Genera n completaciones del mismo prompt en paralelo (ids de tokens).

    Como todas comparten el prompt no hace falta padding: el prefill se hace una vez
    con el prompt repetido y luego se decodifica token a token. Cada completación se
    corta al aparecer una de las paradas o al emitir EOS.
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
        for i, t in enumerate(sig.tolist()):
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
