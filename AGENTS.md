# Instrucciones para agentes

## Principios de diseño

Antes de diseñar, escribir o revisar código, leé [CONSTITUTION.md](CONSTITUTION.md). Sus cinco principios son el criterio de diseño del proyecto:

- Modularidad
- Alta cohesión
- Separación de responsabilidades
- Abstracción
- Bajo acoplamiento

Si un cambio no puede cumplir alguno de los principios, decilo explícitamente y explicá por qué.

## Lenguaje del dominio

El glosario está en [CONTEXT.md](CONTEXT.md) y las decisiones de diseño, en [docs/adr/](docs/adr/). Usá esos términos en el código, los nombres, los docstrings y la conversación, y evitá los que el glosario marca con _Evitar_. Si un término nuevo se resuelve o uno existente cambia, actualizá `CONTEXT.md` en el mismo commit. Para esa tarea está el skill `domain-modeling` (`npx skills add mattpocock/skills@domain-modeling`).

## Antes de commitear

```bash
uv run ruff format imagine evals tests
uv run ruff check imagine evals tests
uv run pytest
```

`tests/test_arquitectura.py` hace cumplir la separación en capas descrita en el README: si una capa importa otra que no le corresponde, falla. Si una capa nueva o una dependencia nueva está justificada, actualizá ahí la tabla `PERMITIDAS` y la del README en el mismo commit.
