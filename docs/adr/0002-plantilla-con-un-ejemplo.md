# La plantilla incluye un ejemplo, a diferencia de DeepSeek-R1-Zero

DeepSeek-R1-Zero usa una plantilla sin ejemplos y el modelo descubre solo el formato. Con un modelo base de 0.8B eso no funciona: sin ejemplo cierra `</imagine>` sólo en ~5 % de los intentos y casi no hay señal para aprender. Con un ejemplo inventado (Elena y la carta) llega a ~90 % a temperatura 0,8. Los modelos entrenados aprendieron con esta plantilla exacta: cambiarla, aunque sea en una palabra, obliga a reentrenar o a medir de nuevo.
