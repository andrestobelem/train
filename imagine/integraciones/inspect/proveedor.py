"""Proveedor `mlx/` para Inspect AI: modelos base de MLX, con o sin adaptadores LoRA.

El nombre del modelo es un repo de Hugging Face o una carpeta de adaptadores del
proyecto (ver `imagine.ejecucion.modelos.resolver`):

    mlx/Qwen/Qwen3.5-0.8B-Base
    mlx/runs/imagine300/paso_00100

Son modelos base: no se aplica plantilla de chat, el texto de los mensajes se pasa
tal cual como prompt.
"""

from inspect_ai.model import (
    ChatMessage,
    GenerateConfig,
    ModelAPI,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolChoice, ToolInfo

from imagine.ejecucion.hilo import en_hilo_mlx, en_hilo_mlx_async
from imagine.ejecucion.lm import muestrear
from imagine.ejecucion.modelos import cargar_por_nombre


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
        self.model, self.tok = en_hilo_mlx(cargar_por_nombre, model_name)

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
        temperatura = config.temperature if config.temperature is not None else 1.0
        salida = muestrear(
            self.model, self.tok, ids, 1, config.max_tokens or 256, temperatura, paradas
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
