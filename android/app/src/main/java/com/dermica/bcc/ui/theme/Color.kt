package com.dermica.bcc.ui.theme

import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.ui.graphics.Color

// Paleta clínica: verde azulado (dermatología) + neutros; contraste pensado para
// lectura prolongada sobre fondo claro y oscuro. Consumidor único: Theme.kt
// (DermicaTheme), que la expone vía MaterialTheme a toda la UI.
// Origen: decisión de diseño del proyecto (tema clínico dermatológico); sin
// colores dinámicos de Android 12+: el esquema debe ser estable en cualquier
// dispositivo. Único matiz semántico: TealDark aclara el primario en modo
// oscuro para mantener contraste sobre superficie oscura.

// Primario modo claro — teal clínico oscuro (0F766E).
private val Teal = Color(0xFF0F766E)
// Primario modo oscuro — mismo teal aclarado (14B8A6) para contraste sobre slate-900.
private val TealDark = Color(0xFF14B8A6)

/** Esquema claro (lightColorScheme). Consumidor: DermicaTheme (isSystemInDarkTheme == false). */
internal val LightColors = lightColorScheme(
    primary = Teal,
    onPrimary = Color.White, // contraste máximo sobre Teal 0F766E
    surface = Color(0xFFF8FAFC), // slate-50: fondo frío, no blanco puro
    onSurface = Color(0xFF0F172A), // slate-900
)

/** Esquema oscuro (darkColorScheme). Consumidor: DermicaTheme (isSystemInDarkTheme == true). */
internal val DarkColors = darkColorScheme(
    primary = TealDark,
    onPrimary = Color(0xFF04211E), // teal-950: texto oscuro sobre el primario claro
    surface = Color(0xFF0F172A), // slate-900
    onSurface = Color(0xFFE2E8F0), // slate-200
)
