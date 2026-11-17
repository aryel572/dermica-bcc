package com.dermica.bcc.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable

/** Tema de la app: claro/oscuro siguiendo el sistema (Material 3, sin paleta dinámica). */
@Composable
fun DermicaTheme(content: @Composable () -> Unit) {
    val colors = if (isSystemInDarkTheme()) DarkColors else LightColors
    MaterialTheme(
        colorScheme = colors,
        typography = DermicaTypography,
        content = content,
    )
}
