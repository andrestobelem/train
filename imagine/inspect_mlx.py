"""Proveedor `mlx/` para Inspect AI: modelos base de MLX, con o sin adaptadores LoRA.

El nombre del modelo es un repo de Hugging Face o una carpeta de adaptadores
guardada por `imagine.entrenar` (su `adapter_config.json` dice cuál es el modelo base):

    mlx/Qwen/Qwen3.5-0.8B-Base
    mlx/runs/imagine300/paso_00100

Son modelos base: no se aplica plantilla de chat, el texto de los mensajes se pasa
tal cual como prompt. Se registra en Inspect con el entry point de pyproject.toml.
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from .modelo import cargar, muestrear

RAIZ = Path(__file__).resolve().parents[1]

# MLX ata sus streams al hilo que los crea: cargar, generar y puntuar tiene que
# pasar siempre por el mismo hilo, que además serializa el uso de la GPU.
_HILO_MLX = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")


def en_hilo_mlx(fn, *args, **kwargs):
    """Corre fn en el hilo de MLX y espera el resultado (desde código sincrónico)."""
    return _HILO_MLX.submit(fn, *args, **kwargs).result()


async def en_hilo_mlx_async(fn, *args):
    """Corre fn en el hilo de MLX sin bloquear el event loop de Inspect."""
    return await asyncio.get_running_loop().run_in_executor(_HILO_MLX, fn, *args)


@modelapi(name="mlx")
class MLXAPI(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args,
    ):
        super().__init__(model_name, base_url, api_key, [], config)
        # Las carpetas relativas se resuelven contra la raíz del proyecto: Inspect
        # cambia el directorio de trabajo al cargar las tareas.
        ruta = Path(model_name)
        if not ruta.is_absolute():
            ruta = RAIZ / ruta
        if (ruta / "adapter_config.json").exists():
            base = json.loads((ruta / "adapter_config.json").read_text())["model"]
            self.model, self.tok = en_hilo_mlx(cargar, str(ruta), base=base)
        else:
            self.model, self.tok = en_hilo_mlx(cargar, base=model_name)

    def max_connections(self) -> int:
        return 1

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        prompt = "\n".join(m.text for m in input)
        return await en_hilo_mlx_async(self._generar, prompt, config)

    def _generar(self, prompt: str, config: GenerateConfig) -> ModelOutput:
        paradas = tuple(config.stop_seqs or ())
        ids = self.tok.encode(prompt)
        salida = muestrear(
            self.model,
            self.tok,
            ids,
            1,
            config.max_tokens or 256,
            config.temperature if config.temperature is not None else 1.0,
            paradas,
        )[0]
        texto = self.tok.decode(salida)
        motivo, parada = "max_tokens", None
        for p in paradas:
            if p in texto:
                texto, motivo, parada = texto.split(p)[0], "stop", p
        if parada is None and salida and salida[-1] == self.tok.eos_token_id:
            motivo = "stop"
        out = ModelOutput.from_content(self.model_name, texto, stop_reason=motivo)
        # Qué parada cortó la salida: distingue cerrar una etiqueta de emitir EOS.
        out.metadata = {"parada": parada}
        out.usage = ModelUsage(
            input_tokens=len(ids),
            output_tokens=len(salida),
            total_tokens=len(ids) + len(salida),
        )
        return out
