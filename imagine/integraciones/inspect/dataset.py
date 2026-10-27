"""El dataset de evaluación en formato Inspect, común a todas las tareas."""

from inspect_ai.dataset import MemoryDataset, Sample

from imagine.datos.repositorio import ejemplos_de_prueba
from imagine.dominio.escritura import prompt


def niebla(n: int) -> MemoryDataset:
    """Ejemplos del libro de prueba (*Niebla*).

    El input es el prompt completo (termina abriendo <imagine>); el target es la
    referencia, y el fragmento queda en la metadata para armar otros prompts.
    """
    return MemoryDataset(
        [
            Sample(
                id=i,
                input=prompt(ejemplo.fragmento),
                target=ejemplo.referencia,
                metadata={"fragmento": ejemplo.fragmento},
            )
            for i, ejemplo in enumerate(ejemplos_de_prueba(n))
        ]
    )
