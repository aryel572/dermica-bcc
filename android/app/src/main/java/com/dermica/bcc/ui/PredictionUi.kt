package com.dermica.bcc.ui

/**
 * Resultado de predicción listo para pintar en ResultScreen.
 *
 * imageBytes se conserva en memoria para el thumbnail; no se persiste en Room
 * (las fotos del usuario nunca salen del dispositivo ni se cachean, S10).
 * ByteArray en un data class: igualdad por referencia, suficiente porque el
 * objeto no se compara ni se serializa (coherente con ponytail: sin copy/equals
 * artesanales que nadie usa).
 */
data class PredictionUi(
    val isBcc: Boolean,
    val confidence: Float, // probabilidad de la clase predicha en [0,1]
    val imageBytes: ByteArray,
    val imageName: String?,
)
