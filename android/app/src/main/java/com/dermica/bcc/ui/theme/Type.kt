package com.dermica.bcc.ui.theme

import androidx.compose.material3.Typography

// Tipografía de la app: por defecto de Material 3 (sin personalización de fuente).
// Consumidor: Theme.kt (DermicaTheme → MaterialTheme.typography en toda la UI).
// Origen: decisión ponytail (ningún requisito pide fuente propia); si el
// producto pide personalización, aquí se define y en Theme.kt no cambia nada.
// Nota: val de solo lectura; MaterialTheme lo expone vía LocalTypography y los
// call sites consumen MaterialTheme.typography.* sin conocer esta constante.
val DermicaTypography = Typography()
