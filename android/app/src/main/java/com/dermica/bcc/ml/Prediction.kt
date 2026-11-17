package com.dermica.bcc.ml

/**
 * Clase binaria del modelo (orden fijo NO_BCC=0, BCC=1, igual que S3/S4).
 *
 * Lugar en el flujo: motor ML (ml). La produce OutputDecoder.decode y la
 * consumen BccPredictor (propaga la etiqueta) y AppRoot/ResultScreen (muestran
 * el rótulo BCC/NO_BCC y el aviso de no sustitución de criterio médico).
 */
enum class BccLabel { NO_BCC, BCC }

/**
 * Resultado de inferencia on-device.
 *
 * Lugar en el flujo: motor ML (ml). Lo produce OutputDecoder.decode (vía
 * BccPredictor) y lo consumen AppRoot (addImagePrediction + navegación al
 * resultado) y ResultScreen (etiqueta, confianza y aviso médico). Salida visible:
 * la card de resultado.
 *
 * @param label — BccLabel: etiqueta decodificada con el umbral de validación (0.5).
 * @param confidence — Float: probabilidad de la clase predicha en [0,1]:
 *        rawProbability si label==BCC, 1−rawProbability si NO_BCC.
 * @param rawProbability — Float: sigmoid cruda de la unidad Dense(1, sigmoid) en [0,1].
 */
data class Prediction(
    val label: BccLabel,
    val confidence: Float,
    val rawProbability: Float,
)
