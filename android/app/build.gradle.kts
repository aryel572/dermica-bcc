import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

// APP_API_KEY se lee de local.properties (gitignored) y se expone vía BuildConfig.
// Si falta queda vacía: la app compila igual y simplemente no puede llamar a la API
// (S10: la clave jamás va al repositorio ni a logs).
val localProps = Properties().apply {
    val localFile = rootProject.file("local.properties")
    if (localFile.exists()) localFile.inputStream().use { load(it) }
}
val appApiKey: String = localProps.getProperty("APP_API_KEY").orEmpty()
    .replace("\\", "\\\\")
    .replace("\"", "\\\"")

android {
    namespace = "com.dermica.bcc"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.dermica.bcc"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
        buildConfigField("String", "APP_API_KEY", "\"$appApiKey\"")
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    // SDK local solo tiene android-35 y build-tools 35: se fija explícitamente.
    buildToolsVersion = "35.0.0"

    // Firma del build release. Las credenciales viven en local.properties (gitignored);
    // el keystore en ~/.android/dermica-release.jks (fuera del repositorio, jamás se commitea).
    // Sin credenciales, el build falla aquí mismo con un mensaje claro.
    fun requiredRelease(key: String): String =
        localProps.getProperty(key) ?: error(
            "Falta $key en android/local.properties; crea la credencial y reintenta el build release"
        )

    signingConfigs {
        create("release") {
            storeFile = file(System.getProperty("user.home") + "/.android/dermica-release.jks")
            storePassword = requiredRelease("RELEASE_STORE_PASSWORD")
            keyAlias = requiredRelease("RELEASE_KEY_ALIAS")
            keyPassword = requiredRelease("RELEASE_KEY_PASSWORD")
        }
    }

    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            isShrinkResources = false
            signingConfig = signingConfigs.getByName("release")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }
}

// Room exporta el schema JSON para auditoría de migraciones futuras (hard rule de storage).
ksp {
    arg("room.schemaLocation", "$projectDir/schemas")
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.06.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.datastore:datastore-preferences:1.1.1")
    // Room: historial persistente de predicciones (S6/S9). KSP en lugar de kapt (Kotlin 2.x).
    implementation("androidx.room:room-runtime:2.8.5")
    implementation("androidx.room:room-ktx:2.8.5")
    ksp("androidx.room:room-compiler:2.8.5")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
    // SAF ACTION_OPEN_DOCUMENT_TREE: DocumentsContract con buen manejo de Uri (S9).
    implementation("androidx.documentfile:documentfile:1.0.1")
    // ViewModel + StateFlow para el orquestador del modo subconjunto (S2/S9).
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    // JSON de referencia en JVM y Android con la misma API (el org.json de Android
    // es un stub en tests unitarios JVM; el artefacto Maven trae la implementación).
    implementation("org.json:json:20240303")
    // LiteRT (sucesor oficial de TFLite): runtime de inferencia on-device (S9).
    // La API Java conserva el paquete org.tensorflow.lite: artefacto renombrado, paquete no.
    implementation("com.google.ai.edge.litert:litert:2.3.2")

    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation("junit:junit:4.13.2")
}

// Nombre del artefacto de salida: APP_DERMICA-debug.apk / APP_DERMICA-release.apk
base {
    archivesName.set("APP_DERMICA")
}
