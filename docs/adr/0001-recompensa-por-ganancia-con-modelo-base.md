# La recompensa es la ganancia medida por el modelo base congelado

Para premiar una imaginación medimos cuánto sube la probabilidad de la referencia según el modelo base congelado, contra una imaginación vacía, en lugar de usar un juez LLM o una similitud de n-gramas con la referencia. Es verificable contra el corpus sin entrenar un modelo de recompensa, por la misma razón por la que DeepSeek-R1 evita las recompensas neuronales. Además la imaginación nunca ve la referencia, así que sólo gana si la anticipa de verdad.

## Opciones consideradas

- **Juez LLM**: mide calidad literaria, pero es caro en una Mac, no es verificable y se presta a que el escritor aprenda a complacer al juez.
- **Similitud de n-gramas (chrF) entre la continuación y la referencia**: es barata, pero premia copiar el vocabulario y no la escena. La seguimos usando sólo para evaluar.

## Consecuencias

La ganancia premia imaginaciones que anticipan, no prosa linda: la calidad de la continuación se mide aparte (tarea de prosa) y se mejora con la selección y el ajuste supervisado. Todos los modelos entrenados dependen de esta definición: cambiarla obliga a reentrenar.
