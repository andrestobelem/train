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
El final del texto previo que ve el escritor: unas 180 palabras como mucho, marcado con «…» si se recortó.
_Evitar_: contexto, prompt, pasaje

**Imaginación**:
Lo que el escritor escribe entre `<imagine>` y `</imagine>` antes de continuar: qué sienten los personajes y cómo sigue la escena.
_Evitar_: imaginado, razonamiento, pensamiento, plan

**Continuación**:
Lo que el escritor escribe después de la imaginación, para el lector.
_Evitar_: respuesta, completación, salida

**Turno**:
Cada intervención del lector o del escritor. Una continuación termina cuando empieza el turno siguiente del lector.

### El corpus

**Libro**:
Una novela en español de dominio público de la que se sacan ejemplos.

**Libro de prueba**:
El libro que se reserva para evaluar y nunca se usa para entrenar: *Niebla*.
_Evitar_: test set, validación

**Referencia**:
Lo que el autor escribió inmediatamente después del fragmento. El escritor nunca la ve.
_Evitar_: continuación del autor, gold, target, texto real

**Ejemplo**:
Un fragmento junto con su referencia, sacados del mismo libro.
_Evitar_: fila, muestra, sample

### Cómo se juzga una imaginación

**Formato válido**:
Una imaginación cierra la etiqueta, no la vuelve a abrir y tiene al menos cinco palabras.

**Ganancia**:
Cuánto mejora la imaginación la probabilidad de la referencia, medida por el modelo base congelado contra una imaginación vacía, en nats por token. Sólo existe si el formato es válido. Se escribe Δ.
_Evitar_: delta, mejora, score

**Imaginación que ayuda**:
Una imaginación con ganancia positiva.

**Recompensa**:
El valor que recibe una imaginación al entrenar: cero sin formato válido; con formato válido, crece con la ganancia.
_Evitar_: reward, puntaje

### Cómo se entrena

**Grupo**:
Las imaginaciones que el escritor produce para un mismo fragmento y que se comparan entre sí para aprender.

**Selección**:
Quedarse, por cada fragmento, con la imaginación de mayor ganancia, siempre que ayude. Un fragmento sin imaginaciones que ayuden se descarta.
_Evitar_: rejection sampling, filtrado

**Ejemplo supervisado**:
Un ejemplo junto con la imaginación elegida en la selección. Enseña a imaginar así y a continuar con la referencia.

**Modelo base**:
El modelo de lenguaje sin entrenar por este proyecto. También es el juez congelado que mide la ganancia.
_Evitar_: modelo de referencia (choca con *referencia*)

**Modelo final**:
El escritor que se usa por defecto: el resultado del aprendizaje por refuerzo y del ajuste supervisado.
