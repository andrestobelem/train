"""Un único hilo para MLX.

MLX ata sus streams al hilo que los crea. Las apps con varios hilos (Inspect, Gradio)
tienen que cargar, generar y puntuar siempre en el mismo hilo, que además serializa
el uso de la GPU. Los scripts de un solo hilo no lo necesitan.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

_HILO_MLX = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def en_hilo_mlx(fn, *args, **kwargs):
    """Corre fn en el hilo de MLX y espera el resultado (desde código sincrónico)."""
    return _HILO_MLX.submit(fn, *args, **kwargs).result()


async def en_hilo_mlx_async(fn, *args):
    """Corre fn en el hilo de MLX sin bloquear el event loop."""
    return await asyncio.get_running_loop().run_in_executor(_HILO_MLX, fn, *args)
