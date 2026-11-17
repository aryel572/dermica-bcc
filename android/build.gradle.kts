// Raíz del monorepo Android: solo declara plugins (apply false); el único módulo es :app.
plugins {
    id("com.android.application") version "8.13.2" apply false
    id("org.jetbrains.kotlin.android") version "2.4.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.4.21" apply false
    // Kotlin 2.4 no tiene KSP publicado aún (tope 2.3.x); 2.3.14 es la versión que resuelve (S6/Room).
    id("com.google.devtools.ksp") version "2.3.14" apply false
}
