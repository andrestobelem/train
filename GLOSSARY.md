# Escritura imaginada

Un modelo de lenguaje aprende a imaginar la escena antes de continuar una novela. Lo que imagina vale tanto como ayude a anticipar lo que el autor escribió de verdad.

## Lenguaje

### La escena de escritura

**Lector**:
El rol que comparte el fragmento y cede el turno.
_Evitar_: usuario, prompt

**Escritor**:
El rol que imagina y después continúa la novela. Lo cumple un modelo.
_Evitar_: asistente, policy

**Fragmento**:
El final del texto previo que ve el escritor, marcado con «…» si se recortó.
_Evitar_: contexto, prompt, pasaje

**Imaginación**:
Lo que el escritor escribe entre `<imagine>` y `</imagine>` antes de continuar: qué sienten los personajes y cómo sigue la escena.
_Evitar_: imaginado, razonamiento, pensamiento, plan

**Continuación**:
Lo que el escritor escribe después de la imaginación, para el lector.
_Evitar_: respuesta, completación, salida

**Turno**:
Cada intervención del lector o del escritor. Los turnos alternan y empieza el lector. Un turno puede quedar vacío: quien no escribió nada pasó, y le toca al otro. Una continuación termina cuando empieza el turno siguiente del lector.

**Historia a cuatro manos**:
Una historia que el lector y el escritor escriben por turnos. En cada turno suyo, el escritor ve como fragmento el final de la historia.
_Evitar_: chat, conversación

### El corpus

**Libro**:
Una novela en español de dominio público de la que se sacan ejemplos.

**Corpus**:
Los libros de entrenamiento más los libros de prueba.

**Libro de entrenamiento**:
Un libro cuyos ejemplos se usan para entrenar al escritor.
_Evitar_: train set

**Libro de prueba**:
Un libro que se reserva para evaluar y nunca se usa para entrenar. Su autor no escribió ningún libro de entrenamiento, así que la prueba mide si el escritor generaliza a otro autor y no si recuerda un estilo ya visto.
_Evitar_: test set, validación

**Referencia**:
Lo que el autor escribió inmediatamente después del fragmento. El escritor nunca la ve.
_Evitar_: continuación del autor, gold, target, texto real

**Ejemplo**:
Un fragmento junto con su referencia, sacados del mismo libro.
_Evitar_: fila, muestra, sample

### Cómo se juzga una imaginación

**Formato válido**:
Una imaginación cierra la etiqueta, no la vuelve a abrir y no está casi vacía.

**Ganancia**:
Cuánto mejora la imaginación la probabilidad de la referencia según el juez, frente a no imaginar nada. Sólo existe si el formato es válido. Se escribe Δ.
_Evitar_: delta, mejora, score

**Juez**:
El modelo que mide la ganancia. Está congelado: no aprende de lo que juzga. Hoy es el modelo base.
_Evitar_: modelo de referencia, evaluador

**Imaginación que ayuda**:
Una imaginación con ganancia positiva.

**Recompensa**:
El valor que recibe una imaginación al entrenar: cero sin formato válido; con formato válido, crece con la ganancia.
_Evitar_: reward, puntaje

### Cómo se evalúa la continuación

**Fluidez**:
Qué tan esperable le resulta al juez la continuación, dado sólo el fragmento y no la imaginación. Por eso una continuación que sigue fielmente a la imaginación puede ser buena y poco fluida.

**Parecido con el autor**:
Cuánto se parece la continuación a la referencia, letra por letra en secuencias cortas. No mide si la escena es la misma, sólo si se escribe parecido.
_Evitar_: chrF, similitud, score

**Letras ajenas**:
Las letras de la continuación que no son del alfabeto latino, como griego o chino: un síntoma de que el escritor se desvió del español.

**Intento**:
Cada vez que, al evaluar, el escritor imagina para el mismo fragmento.
_Evitar_: muestra, época

### Cómo se entrena

**Grupo**:
Las imaginaciones que el escritor produce para un mismo fragmento y que se comparan entre sí para aprender. La de mayor recompensa es la **mejor imaginación** del grupo; si ninguna tiene formato válido, el grupo no tiene mejor imaginación.

**Selección**:
Quedarse, por cada fragmento, con la imaginación de mayor ganancia, siempre que ayude. Un fragmento sin imaginaciones que ayuden se descarta.
_Evitar_: rejection sampling, filtrado

**Ejemplo supervisado**:
Un ejemplo junto con la imaginación elegida en la selección. Enseña a imaginar así y a continuar con la referencia.

**Modelo base**:
El modelo de lenguaje sin entrenar por este proyecto. Es el punto de partida del escritor y el ancla de la que no debe alejarse demasiado al aprender.
_Evitar_: modelo de referencia (choca con *referencia*)

**Modelo final**:
El escritor que se usa por defecto: el resultado del aprendizaje por refuerzo y del ajuste supervisado.
