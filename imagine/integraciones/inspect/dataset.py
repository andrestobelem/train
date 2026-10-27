"""El dataset de evaluación en formato Inspect, común a todas las tareas."""

from inspect_ai.dataset import MemoryDataset, Sample

from imagine.datos.archivos import TEST, leer_jsonl
from imagine.dominio.escritura import prompt


def niebla(n: int) -> MemoryDataset:
    """Fragmentos de *Niebla*, el libro que el modelo no vio al entrenar.

    El input es el prompt completo (termina abriendo <imagine>); el target es lo que
    escribió el autor, y el fragmento queda en la metadata para armar otros prompts.
    """
    return MemoryDataset(
        [
            Sample(
                id=i,
                input=prompt(f["contexto"]),
                target=f["referencia"],
                metadata={"fragmento": f["contexto"]},
            )
            for i, f in enumerate(leer_jsonl(TEST, n))
        ]
    )
