"""Opciones comunes a los flows de este directorio (Inspect Flow lo incluye solo)."""

from inspect_flow import FlowOptions, FlowSpec

FlowSpec(
    # MLX usa una sola GPU: de a una tarea y de a una muestra.
    options=FlowOptions(max_tasks=1, max_samples=1, display="plain", log_dir_allow_dirty=True),
)
