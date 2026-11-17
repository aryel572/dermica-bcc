# ANDROID — APP_DERMICA (BCC)

Aplicación Android nativa (Kotlin + Jetpack Compose) para **apoyo al tamizaje del carcinoma basocelular (BCC)** en imágenes dermatoscópicas, con **inferencia on-device** mediante el modelo release 1.0.0 (MobileNetV3-Small, `model-float16.tflite`, LiteRT).

**Esta aplicación es una herramienta de apoyo a la decisión médica. No sustituye el diagnóstico profesional.**

## Stack

- Kotlin 2.4 · Gradle 8.13 (wrapper) · AGP 8.13 · compileSdk 35 / minSdk 26
- Jetpack Compose Material 3 · Navigation manual (2 destinos) · Room 2.8 (historial)
- `com.google.ai.edge.litert` (TensorFlow Lite renombrado) para inferencia local
- `DataStore` (flags) · `HttpURLConnection` para actualización del modelo (sin otras deps de red)

## Build y tests

```bash
cd android
./gradlew :app:assembleDebug        # APK debug
./gradlew :app:testDebugUnitTest    # tests JVM (59/59)
./gradlew :app:assembleRelease      # APK release firmado (minify R8); credenciales en local.properties
```

Requiere Android SDK con `android-35` + `build-tools 35.0.0` (o ajusta `buildToolsVersion`) y JDK 21.

## Modelo (baseline offline + actualización)

- **Base empaquetada**: `app/src/main/assets/model-float16.tflite` (1 945 356 B, SHA-256 `d62c401c…`). En el primer arranque se copia a almacenamiento interno y se **verifica por SHA-256** antes de usarse.
- **Actualización**: la app consulta el backend FastAPI (`/model/version` → `/model/latest` con `Authorization: Bearer` → `/model/download`). El binario descargado se **verifica (SHA-256 + tamaño) antes de tocar el modelo instalado** y se escribe con rename atómico (`model.tflite.download` → `model.tflite`).
- **Eliminar modelo** (sección Configuración): borra la copia interna y desactiva su uso hasta descargar de nuevo.
- El modelo **nunca sale del dispositivo** hacia ningún servicio propio; la inferencia es 100 % local.

## Configuración

Copia `local.properties.example` o crea `android/local.properties` (gitignored):

```properties
APP_API_KEY=tu_clave_aquí
```

Sin clave la app compila y predice normal (con el modelo base); **solo** los botones de descarga/actualización permanecen deshabilitados con explicación.

## Privacidad

- El historial es local (Room en almacenamiento interno); ninguna foto se sube a red alguna.
- No hay telemetría ni analítica.
- El modelo descargado siempre pasa verificación de integridad antes de cargarse.

## Estado del modelo en la app

Configuración → muestra versión instalada, cuantización (float16) y SHA parcial; botones **Eliminar modelo** (con confirmación) y **Descargar modelo** / **Buscar actualización** (con aviso de arranque en frío del plan gratuito: hasta ~60 s).
