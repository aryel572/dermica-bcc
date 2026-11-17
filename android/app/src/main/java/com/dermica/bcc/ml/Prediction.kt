package com.dermica.bcc.ml

/** Clase binaria del modelo (orden fijo NO_BCC=0, BCC=1, igual que S3/S4). */
enum class BccLabel { NO_BCC, BCC }

/**
 * Resultado de inferencia on-device.
 *
 * @param label etiqueta decodificada con el umbral de validación (0.5)
 * @param confidence probabilidad de la clase predicha en [0,1]
 * @param rawProbability sigmoid cruda de la unidad Dense(1, sigmoid) en [0,1]
 */
data class Prediction(
    val label: BccLabel,
    val confidence: Float,
    val rawProbability: Float,
)
