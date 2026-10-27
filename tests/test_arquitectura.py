"""Las dependencias entre capas van sólo hacia abajo (CONSTITUTION.md: separación de
responsabilidades y bajo acoplamiento). Si este test falla, una capa importó algo que
no le corresponde."""

import ast
from pathlib import Path

import pytest

PAQUETE = Path(__file__).resolve().parents[1] / "imagine"

PERMITIDAS = {
    "dominio": set(),
    "persistencia": set(),
    "datos": {"dominio"},
    "ejecucion": {"dominio", "datos", "persistencia"},
    "entrenamiento": {"dominio", "datos", "persistencia", "ejecucion"},
    "integraciones": {"dominio", "datos", "persistencia", "ejecucion"},
    "interfaz": {"dominio", "datos", "persistencia", "ejecucion"},
}

# Librerías de ejecución que el dominio no puede tocar.
PESADAS = {"mlx", "mlx_lm", "inspect_ai", "inspect_flow", "gradio", "sacrebleu"}


def _imports(archivo: Path) -> set[str]:
    nombres = set()
    for nodo in ast.walk(ast.parse(archivo.read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.Import):
            nombres |= {a.name for a in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.level == 0:
            nombres.add(nodo.module)
    return nombres


def _capas_importadas(archivo: Path) -> set[str]:
    return {n.split(".")[1] for n in _imports(archivo) if n.startswith("imagine.") and n.count(".")}


@pytest.mark.parametrize("capa", sorted(PERMITIDAS))
def test_cada_capa_importa_solo_capas_inferiores(capa):
    for archivo in (PAQUETE / capa).rglob("*.py"):
        ajenas = _capas_importadas(archivo) - PERMITIDAS[capa] - {capa}
        assert not ajenas, f"{archivo.relative_to(PAQUETE)} importa {sorted(ajenas)}"


def test_el_dominio_es_python_puro():
    for archivo in (PAQUETE / "dominio").rglob("*.py"):
        raices = {n.split(".")[0] for n in _imports(archivo)}
        assert not raices & PESADAS, f"{archivo.name} importa {sorted(raices & PESADAS)}"


def test_todas_las_capas_estan_declaradas():
    paquetes = {p.name for p in PAQUETE.iterdir() if (p / "__init__.py").exists()}
    assert paquetes == set(PERMITIDAS)
