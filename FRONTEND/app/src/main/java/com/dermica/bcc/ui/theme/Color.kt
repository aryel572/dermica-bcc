package com.dermica.bcc.ui.theme

import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color

// Paleta clínica: verde azulado (dermatología) + neutros; contraste pensado para
// lectura prolongada sobre fondo claro y oscuro.
private val Teal = Color(0xFF0F766E)
private val TealDark = Color(0xFF14B8A6)

internal val LightColors = lightColorScheme(
    primary = Teal,
    onPrimary = Color.White,
    surface = Color(0xFFF8FAFC),
    onSurface = Color(0xFF0F172A),
)

internal val DarkColors = darkColorScheme(
    primary = TealDark,
    onPrimary = Color(0xFF04211E),
    surface = Color(0xFF0F172A),
    onSurface = Color(0xFFE2E8F0),
)
