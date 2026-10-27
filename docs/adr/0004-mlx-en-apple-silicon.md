# Se entrena y se sirve con MLX en Apple Silicon

Elegimos MLX en lugar de PyTorch para entrenar y generar en una Mac M2 Max, porque está hecho para la memoria unificada de Apple Silicon y mlx-lm ya soporta Qwen3.5 y LoRA. No comparamos su velocidad contra PyTorch sobre MPS. El costo es el encierro: el proveedor de Inspect, los adaptadores y el código de entrenamiento son de MLX. Correr en Linux, o publicar en un Space de Hugging Face, requiere fusionar los adaptadores y convertir los pesos a `transformers`.
