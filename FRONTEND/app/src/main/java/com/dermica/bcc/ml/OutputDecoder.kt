package com.dermica.bcc.ml

/**
 * Decodificación de la salida del modelo a una clase binaria.
 *
 * Fidelidad con la evaluación (S4): el modelo termina en Dense(1, sigmoid);
 * evaluate.py y metrics.confusion_at usan umbral 0.5 sobre el sigmoid crudo
 * (preds = y_score >= 0.5). La clase positiva (1) es BCC; NO_BCC=0, BCC=1.
 */
object OutputDecoder {

    /** Umbral de decisión usado en validación/test: corte neutral sobre la sigmoide. */
    const val THRESHOLD = 0.5f

    /**
     * Decodifica la salida cruda del Interpreter.
     *
     * Parámetros:
     *   rawOutput — sigmoid 1 unidad (layout [1], tal como evaluate.py hace reshape(-1)).
     * Retorna: Prediction con etiqueta, confianza de la clase predicha y probabilidad cruda.
     * Lanza IllegalArgumentException si la salida no tiene exactamente 1 unidad.
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
