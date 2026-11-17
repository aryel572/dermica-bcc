package com.dermica.bcc.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

/**
 * Tema de la app: claro/oscuro siguiendo el sistema (Material 3, sin paleta
 * dinámica). Único consumidor: MainActivity (setContent). No expone API: toda
 * la UI toma color y tipografía vía MaterialTheme.
 *
 * @param content jerarquía composable bajo el tema (en la práctica: AppRoot).
 *
 * Decide el esquema en cada recomposición con isSystemInDarkTheme(); no hace
 * falta remember porque los ColorScheme son objetos estables e inmutables.
 * DermicaTypography se inyecta aparte (Type.kt) porque no depende del modo.
 */
@Composable
fun DermicaTheme(content: @Composable () -> Unit) {
    val colors = if (isSystemInDarkTheme()) DarkColors else LightColors
    MaterialTheme(
        colorScheme = colors,
        typography = DermicaTypography,
        content = content,
    )
}
