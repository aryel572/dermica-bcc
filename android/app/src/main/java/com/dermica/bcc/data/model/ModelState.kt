package com.dermica.bcc.data.model

/**
 * Estado del modelo empaquetado/interno (S1).
 *
 * Lugar en el flujo: datos (data/model). Lo publica ModelManager como
 * StateFlow<ModelState> y lo consume ui/AppRoot para gobernar el CTA «Imagen
 * única» de la pantalla principal (solo habilitado en READY): decide cuándo
 * crear/liberar BccPredictor. También lo lee SettingsViewModel/SettingsScreen
 * para mostrar el estado y las acciones disponibles del modelo (S7).
 *
 * La UI observa este enum y nunca inventa un estado intermedio: todo transita
 * NO_MODEL → COPYING → READY|FAILED. No se persiste: se recalcula en cada
 * arranque verificando el archivo interno.
 */
enum class ModelState {
    /** Sin modelo interno (borrado por el usuario o aún no restaurado/descargado). */
    NO_MODEL,

    /** Copiando el asset o verificando el SHA-256; la inferencia aún no es posible. */
    COPYING,

    /** Modelo interno íntegro (SHA-256 verificado); la inferencia puede ejecutarse. */
    READY,

    /**
     * Copia interna corrupta o fallo de I/O; la app no predice hasta restaurar/descargar.
     *
     * La UI lo muestra como estado de error con salida clara («Restaurar modelo")
     * en Settings; nunca se intenta inferencia con un archivo dañado.
     */
    FAILED,
}
