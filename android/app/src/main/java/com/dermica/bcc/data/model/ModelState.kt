package com.dermica.bcc.data.model

/**
 * Estado del modelo empaquetado/interno (S1). La UI observa este enum y nunca
 * inventa un estado intermedio: todo transita NO_MODEL → COPYING → READY|FAILED.
 */
enum class ModelState {
    /** Sin modelo interno (borrado por el usuario o aún no restaurado/descargado). */
    NO_MODEL,

    /** Copiando el asset o verificando el SHA-256; la inferencia aún no es posible. */
    COPYING,

    /** Modelo interno íntegro (SHA-256 verificado); la inferencia puede ejecutarse. */
    READY,

    /** Copia interna corrupta o fallo de I/O; la app no predice hasta restaurar/descargar. */
    FAILED,
}
