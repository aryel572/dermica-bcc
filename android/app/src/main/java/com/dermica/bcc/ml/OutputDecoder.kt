package com.dermica.bcc.ml

/**
 * Decodificación de la salida del modelo a una clase binaria.
 *
 * Lugar en el flujo: motor ML (ml), puro JVM. Lo consume BccPredictor tras
 * interpreter.run y lo cubre OutputDecoderTest en JVM. También lo contemplan
 * las métricas de evaluación (S4): el mismo umbral 0.5 que usa el pipeline al
 * calcular métricas de validación/test.
 *
 * Fidelidad con la evaluación (S4): el modelo termina en Dense(1, sigmoid);
 * evaluate.py y metrics.confusion_at usan umbral 0.5 sobre el sigmoid crudo
 * (preds = y_score >= 0.5). La clase positiva (1) es BCC; NO_BCC=0, BCC=1.
 */
object OutputDecoder {

    /**
     * Umbral de decisión usado en validación/test: corte neutral sobre la sigmoide.
     *
     * Origen: contrato de evaluación (S4) — el mismo umbral con el que se midió
     * el release 1.0.0; no se toca sin re-validar el modelo. Consumidor: decode.
     */
    const val THRESHOLD = 0.5f

    /**
     * Decodifica la salida cruda del Interpreter.
     *
     * @param rawOutput — FloatArray: sigmoid de 1 unidad (layout [1], tal como
     *        evaluate.py hace reshape(-1)); debe tener exactamente 1 elemento.
     * @return Prediction con etiqueta, confianza de la clase predicha y
     *         probabilidad cruda sigmoid en [0,1].
     * @throws IllegalArgumentException si la salida no tiene exactamente 1 unidad.
     *
     * Efectos: ninguno; mapeo puro.
     */
    fun decode(rawOutput: FloatArray): Prediction {
        require(rawOutput.size == 1) {
            "Se esperaba salida sigmoid de 1 unidad (Dense(1, sigmoid)), recibido ${rawOutput.size}"
        }
        val probability = rawOutput[0].coerceIn(0f, 1f)
        val label = if (probability >= THRESHOLD) BccLabel.BCC else BccLabel.NO_BCC
        val confidence = if (label == BccLabel.BCC) probability else 1f - probability
        return Prediction(label = label, confidence = confidence, rawProbability = probability)
    }
}
