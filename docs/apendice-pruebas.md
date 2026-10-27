# Apéndice: el modelo, puesto a prueba

Esta página es para leer, no para instalar nada. Muestra qué hace el modelo con textos reales y cómo sabemos si mejoró. No hace falta saber de programación ni de inteligencia artificial.

## Primero, tres ideas

**1. Un modelo de lenguaje es un completador de textos.** Le das el principio de un texto y escribe cómo sigue, palabra por palabra. Aprendió leyendo muchísimo texto, igual que un lector voraz que, de tanto leer, sabe qué palabra suele venir después de otra. El nuestro es chico: unas 800 millones de «perillas», contra cientos de miles de millones de los modelos grandes como ChatGPT. Por eso escribe como un aprendiz.

**2. Le enseñamos a tomar una nota antes de escribir.** Antes de continuar el fragmento, el modelo escribe una nota corta para sí mismo sobre qué sienten los personajes y qué va a pasar. La llamamos **imaginación**, y va entre dos marcas: `<imagine>` y `</imagine>`. Es como el escritor que, antes de seguir el capítulo, anota en el margen «acá ella le confiesa todo».

**3. ¿Cómo se sabe si una nota es buena?** Con un juego de adivinanzas. Tomamos una novela, la cortamos en un punto y tapamos lo que sigue. El modelo escribe su nota sin ver lo tapado. Después, un **juez**, otro modelo que no aprende nada y sólo mide, lee la página tapada dos veces: una después de la nota y otra sin ninguna nota. Y responde una sola pregunta: **¿con la nota, la página real te sorprendió menos?**

- Si la nota hizo que lo que escribió el autor resultara menos sorprendente, la nota **ayudó**: anticipó algo de lo que de verdad pasa.
- Si lo hizo más sorprendente, la nota **despistó**: imaginó algo que no pasa.

Cuando entrenamos, el modelo recibe un premio cada vez que su nota ayuda. Después de miles de rondas del juego, aprende a tomar notas que ayudan más. Nadie le dice qué es una buena nota: lo descubre jugando. Es la misma idea con la que DeepSeek-R1 aprendió a pensar antes de resolver problemas de matemática.

## Dónde se lo prueba

Entrenamos al modelo con *Don Quijote*, de Cervantes, y *Fortunata y Jacinta*, de Galdós. Lo probamos con *Niebla*, de Unamuno, un libro que nunca vio y de un autor que no estaba en el entrenamiento. Así sabemos que no está repitiendo de memoria.

Comparamos dos versiones:

- **El modelo sin entrenar**: el completador original, al que sólo le pedimos que tome la nota.
- **El modelo entrenado**: el mismo, después de jugar miles de rondas.

## El resultado en números

Con 40 fragmentos de *Niebla*, cuatro intentos cada uno:

| | Sin entrenar | Entrenado |
|---|---|---|
| Escribe la nota bien cerrada, con las dos marcas | 84 de cada 100 | 99 de cada 100 |
| La nota ayuda a anticipar lo que escribió Unamuno | 28 de cada 100 | 77 de cada 100 |

En pocas palabras: **antes, imaginar lo despistaba casi siempre; ahora ayuda en tres de cada cuatro casos.**

## Cuatro pruebas, una por una

Los cuatro fragmentos se eligieron **al azar** entre los 200 de *Niebla*, no por ser los que mejor salieron. Las respuestas son exactamente las que escribió cada modelo, con errores incluidos. Los fragmentos están recortados al final: el modelo vio un poco más de texto, hasta unas 180 palabras.

Para cada prueba vas a ver:

1. el final del fragmento que leyó el modelo;
2. lo que de verdad escribió Unamuno después, tapado para el modelo;
3. la nota y la continuación de cada versión;
4. el veredicto del juez sobre la nota.

---

### Prueba 1: Unamuno discute con su personaje

> (…) —Bueno, dejémonos de esos sentires y vamos a otra cosa. Cuando un hombre dormido e inerte en la cama sueña algo, ¿qué es lo que más existe, él como conciencia que sueña o su sueño?
>
> —¿Y si sueña que existe él mismo, el soñador?—le repliqué a mi vez.

**Lo que escribió Unamuno (tapado):**

> —En ese caso, amigo don Miguel, le pregunté yo a mi vez, ¿de qué manera existe él, como soñador que se sueña, o como soñado por sí mismo? Y fíjese, además, en que al admitir esta discusión conmigo me reconoce ya existencia independiente de sí.

**Sin entrenar.** Nota: *«Elena no sabe si un sueño es real o no. Su mente, como una casa que la da hoy, se queda aún en la época de los sueños, no despierta. (…)»*

> —¿Y si no soy yo?—la pregunta se distendió entre sus palabras.

Veredicto: **despistó bastante** (−0,099). Inventó a una «Elena» que no existe en *Niebla*. La sacó del ejemplo que le mostramos para explicarle el juego, que tiene una Elena. Copió el nombre del ejemplo en lugar de leer la escena.

**Entrenado.** Nota: *«La respuesta es que he olvidado de haberse de él, y que mi razón es que el sueño me lo ha dejado.»*

> —Pues es como que no duerme usted, porque en esos momentos de sueño usted puede estar dormido en la misma cama que él, y no se lo importa.
>
> —No se lo importa—replicó don Quijote—, y por lo demás, don Quijote no duerme, ni siquiera cuando está en un lecho sin cama ni en un lomo sin almohada.

Veredicto: **despistó un poco** (−0,040). Despista menos que la nota sin entrenar. La nota es confusa, pero se queda en el tema: el sueño y quién responde. En la continuación se le cuela don Quijote, porque lo leyó mucho al entrenar y en el fragmento se lo nombra.

---

### Prueba 2: Augusto se duerme pensando en Eugenia

> (…) ¡Oh, Eugenia, mi Eugenia, flor de mi aburrimiento vital e inconsciente, asísteme en mis sueños, sueña en mí y conmigo!»
>
> Y quedóse dormido.
>
> V

**Lo que escribió Unamuno (tapado):**

> Cruzaba las nubes, águila refulgente, con las poderosas alas perladas de rocío, fijos los ojos de presa en la niebla solar, dormido el corazón en dulce aburrimiento al amparo del pecho forjado en tempestades (…)

**Sin entrenar.** Nota: *«La familia se ha tornado un espectador. Eugenia no tiene amigos, solo tiene una hermana, y ella no sabe cómo tratarla.»*

> —¿Tú qué haces, Eugenia? —preguntó padres. —¿Qué haces tú cuando la sientes tan... triste?

Veredicto: **despistó bastante** (−0,091). Inventa una hermana y una familia que no aparecen.

**Entrenado.** Nota: *«Eugenia se había de sus labios, y no había de haberse de otra manera; y, pues el silencio les había de cumplir, no había de haberse de más.»*

> «Oye, mimucha, aquí te veo en casa mía.
>
> —No, hombre—respondió ella—, aquí te veo.»
>
> «¿Pero qué, ¿cómo?
>
> —¿No lo sabes? (…)
>
> —Yo he pensado—dijo Eugenia—, que no me hablas de la casa, sino de la... de la... de tu vida.

Veredicto: **despistó un poco** (−0,031). Ninguno de los dos podía adivinar el salto de Unamuno: empieza un capítulo nuevo con un sueño en el que Augusto es un águila. Es una prueba muy difícil. Aun así, el entrenado despista menos. Su continuación, al menos, se queda con Augusto y Eugenia, y no inventa personajes nuevos.

---

### Prueba 3: «No eres más que un personaje de novela»

> (…) —No, no existes más que como ente de ficción; no eres, pobre Augusto, más que un producto de mi fantasía y de las de aquellos de mis lectores que lean el relato que de tus fingidas venturas y malandanzas he escrito yo; tú no eres más que un personaje de novela, o de nivola, o como quieras llamarle. Ya sabes, pues, tu secreto.

**Lo que escribió Unamuno (tapado):**

> Al oir esto quedóse el pobre hombre mirándome un rato con una de esas miradas perforadoras que parecen atravesar la mira e ir más allá, miró luego un momento a mi retrato al óleo que preside a mis libros, le volvió el color y el aliento, fué recobrándose (…)

**Sin entrenar.** Nota: *«Augusto no respiró. La llave que llevaba en el bolsillo de su camisa cayó; sus dedos, que habían traído el cabello, se enroscaron en el frasco.»* Y, sin cerrar la nota, siguió escribiendo diálogo hasta inventar otra ronda del juego.

> —¿Quién es el que yo no encuentro?—preguntó, con voz que parecía una continuación.

Veredicto: **no cuenta**. Nunca escribió la marca de cierre, `</imagine>`, así que no se sabe dónde termina la nota y dónde empieza la novela. Es la falla más común del modelo sin entrenar. Lo curioso es que la primera frase de su nota, «Augusto no respiró», iba bien encaminada: Unamuno describe justamente la reacción física de Augusto.

**Entrenado.** Nota: *«Augusto se había de serlo, pero ya había de serlo, porque se había hecho. (…)»*

> —No es que yo no sea, señor, ni que yo sea. Pero como son cosas tan preciosas, que no se pueden perder, ¡a lo mejor!—dijo con una cualidad de roco que no se le había de dar.
>
> —No te moleste—le contestó—, porque no se te han escapado de la puerta si te has de abrir. (…)

Veredicto: **ayudó un poco** (+0,013). La nota es casi un trabalenguas, pero gira en torno a Augusto y a su ser o no ser, que es lo que sigue en la novela. La continuación tiene una idea linda, «No es que yo no sea, señor, ni que yo sea», aunque después se pierde.

---

### Prueba 4: Augusto se sacrifica por Eugenia

> (…) —¿Usted, don Augusto, usted?
>
> —¡Yo, sí, yo, señora! ¡Estoy dispuesto a sacrificarme por la felicidad de Eugenia, de su sobrina, porque mi felicidad consiste en que ella sea feliz!

**Lo que escribió Unamuno (tapado):**

> —¡Bravo!—exclamó el tío—¡bravo! ¡bravo! ¡He aquí un héroe! ¡he aquí un anarquista... místico!
>
> —¿Anarquista?—dijo Augusto. (…)

**Sin entrenar.** Nota: *«Eugenia está embarazada de un hijo, y sigue separando a su hijo de su madre. El tío Augusto la ayuda a encontrar el lugar de acogida que ella merece, pero Eugenia no puede aceptarlo.»*

> —¡Esa es mi libertad! ¡De manera que nos sintamos libres, y nos dijamos que no podemos todo lo que queramos! (…) —¡Y yo no tengo ninguna pena en el hecho de que soy madre! ¡Sé que soy madre! (…)

Veredicto: **casi no cambió nada** (+0,001). Inventa un embarazo y confunde a Augusto con el tío.

**Entrenado.** Nota: *«Augusto se había de dar por hecho, y no tenía por qué decir más. Su palabra había de serse de aquel caso, y se había de su parecer.»*

> —¡Yo, sí, yo, señora! ¡Yo, sí, yo, señora!
>
> —¿Pero, don Augusto, ¿cuántas veces se me ha de dar por hecho?

Veredicto: **ayudó bastante** (+0,119). Es la mejor de las cuatro. La nota anticipa que Augusto ya dijo lo que tenía que decir y que la escena sigue con otro que le responde. Eso pasa en la novela: el tío lo aplaude y lo llama héroe.

---

## Lo que muestran estas pruebas

**El modelo entrenado aprendió a no despistar.** En las tres pruebas que se pueden comparar, su nota le dejó al juez la página real menos sorprendente que la nota del modelo sin entrenar. En la otra, el modelo sin entrenar ni siquiera cerró la nota. El sin entrenar inventaba datos concretos y falsos: una Elena, una hermana, un embarazo. Cuando la novela no va por ahí, eso despista mucho.

**Pero aprendió una mañana.** Sus notas suenan raras: «se había de», «había de haberse», una y otra vez. Le quedó un tic. Como el premio sólo mira si la nota ayuda a anticipar la página y no si se lee lindo, descubrió que una nota vaga, con el nombre del personaje y algo del tono, rara vez despista. Es como un estudiante que aprende a contestar el examen sin arriesgar: no se equivoca, pero tampoco dice mucho. Es un fenómeno conocido al entrenar con premios: el modelo aprende exactamente lo que se premia, no lo que uno tenía en mente.

**Y sigue escribiendo como un aprendiz.** Las continuaciones arman frases bien, tienen el tono de un diálogo del siglo XIX y a veces una idea buena, pero pierden el hilo en pocas líneas. Con 800 millones de «perillas» no da para más. Lo que queríamos ver era otra cosa: si la idea de premiar las notas que anticipan la novela funciona. Funciona, con sus mañas.

## Para probarlo uno mismo

Hace falta una Mac moderna (con chip M1 o posterior) y que alguien la deje instalada siguiendo [correr-el-modelo.md](correr-el-modelo.md). Después hay dos maneras:

- **En el navegador**: se abre una página donde se pega un fragmento y se aprieta «Continuar». Muestra la nota y la continuación por separado.
- **A cuatro manos**: vos escribís un párrafo, el modelo escribe el siguiente, y así se arma una historia entre los dos.
