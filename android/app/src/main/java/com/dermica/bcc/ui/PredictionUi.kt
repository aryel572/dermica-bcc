package com.dermica.bcc.ui

/**
 * Resultado de predicción listo para pintar en ResultScreen. Lo produce
 * AppRoot.onImageChosen (única vía: inferencia con BccPredictor + persistencia
 * en HistoryRepository en el mismo bloque) y lo consume AppRoute.RESULT →
 * ResultScreen.
 *
 * @param isBcc veredicto booleano del motor; la UI lo traduce a texto de veredicto.
 * @param confidence probabilidad de la clase predicha en [0,1] (no la sigmoid cruda).
 * @param imageBytes se conserva en memoria para el thumbnail; no se persiste en Room
 *        (las fotos del usuario nunca salen del dispositivo ni se cachean, S10).
 * @param imageName nombre visible del archivo (o null si el proveedor SAF no lo da).
 *
 * ByteArray en un data class: igualdad por referencia, suficiente porque el
 * objeto no se compara ni se serializa (coherente con ponytail: sin copy/equals
 * artesanales que nadie usa).
 */
data class PredictionUi(
    val isBcc: Boolean,
    /** Probabilidad de la clase predicha en [0,1]; la UI la muestra como % con un decimal. */
    val confidence: Float,
    val imageBytes: ByteArray,
    val imageName: String?,
)
