# ADR 0003 — Modelo base empaquetado en la app

- **Estado**: Aceptada (decisión tomada en S5; registrada en Fase 0)
- **Fecha**: 2026-11-17
- **Alcance**: `android/` (hoy `FRONTEND/`)

## Contexto

La app debe funcionar desde el primer arranque, sin red y sin configuración del usuario (el tamizaje no puede depender de que el servidor esté disponible ni de que el usuario descargue nada). A la vez, el proyecto necesita poder actualizar el modelo en el futuro sin reinstalar la app desde cero.

Alternativas descartadas: descargar el modelo siempre al primer arranque (falla sin red; mala primera impresión y mal clínicamente); flag de "modelo de demostración" separado del real (duplica rutas de código, invita a olvidarse del camino real).

## Decisión

El **modelo base va empaquetado** como asset de la app (`model-float16.tflite`, 1 945 356 B, SHA-256 `d62c401c…`). En el primer arranque se copia a almacenamiento interno y se **verifica por SHA-256 antes de usarse**. La actualización es **opcional y por el mismo backend real**: `/model/version` → `/model/latest` → `/model/download`, con verificación de **SHA-256 + tamaño antes de reemplazar** y escritura atómica (`model.tflite.download` → `model.tflite`). Se eliminó el flag que fingía la descarga: la app prueba la ruta real contra el backend real, y la sección Configuración permite eliminar el modelo descargado (la app vuelve al estado "sin modelo descargado" con el base empaquetado siempre disponible tras reinstalación).

## Consecuencias

- Funciona offline desde el primer arranque; cero configuración del usuario para el caso base.
- Una sola ruta de código para usar el modelo: siempre verificado por SHA-256, venga de donde venga.
- La actualización real se ejercita y verifica en producción desde el día 1 (nada de código muerto de descarga).
- Costo: el APK pesa ~22 MB (aceptable en el dominio; sin binarios adicionales).
- La app nunca queda sin modelo usable por culpa de una actualización corrupta: el reemplazo es atómico y verificado.
