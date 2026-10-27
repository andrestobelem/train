"""Catálogo de los modelos entrenados del proyecto: dónde quedó cada uno.

Las rutas son relativas a la raíz del proyecto y sirven como nombre de modelo en
todo el código (ver `imagine.ejecucion.modelos.resolver`).
"""

# Corrida de GRPO de 300 pasos, con un checkpoint cada 50 en paso_NNNNN/.
CORRIDA_GRPO = "runs/imagine300"

# Checkpoint de RL del que parte el SFT (paso 250: de los mejores en Niebla).
RL = f"{CORRIDA_GRPO}/paso_00250"

# Modelo final: RL + selección + ajuste supervisado, época 1 (la de menor val loss).
FINAL = "runs/sft/epoca_1"

# Dónde está publicado el modelo final (repo privado de Hugging Face).
REPO_FINAL = "andrestobelem/imagine-qwen3.5-0.8b"

# Salida por defecto de cada entrenamiento.
SALIDA_GRPO = "runs/imagine"
SALIDA_SFT = "runs/sft"
