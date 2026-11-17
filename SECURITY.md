# SECURITY — APP_DERMICA

## Resumen

- **Ningún secreto vive en el repositorio.** Todo lo sensible está en `.env` o `local.properties` (ambos gitignored) o en variables de entorno del servicio desplegado.
- El repo nunca contiene imágenes ISIC ni binarios de datos.
- Las imágenes de la app **nunca salen del dispositivo**; la app no sube fotos a ningún servidor.

## Dónde vive cada secreto

| Secreto | Ubicación | Acceso / alcance | Rotación |
|---------|-----------|------------------|----------|
| `APP_API_KEY` | `local.properties` (gitignored) → expuesta a la app vía `BuildConfig`; copia servidor en variables de entorno del servicio en Render | Bearer sobre `/model/latest` y `/model/download` únicamente (el resto de la API es público o no usa clave) | Generar clave nueva en el backend, actualizar el servicio en Render y `local.properties`, recompilar/redistribuir la app, verificar, revocar la anterior |
| `RELEASE_STORE_PASSWORD` / `RELEASE_KEY_PASSWORD` | `local.properties` (gitignored) | Firma del APK de release | Cambiar junto con el keystore si se compromete (ver abajo) |
| Keystore de release (`dermica-release.jks`) | `~/.android/dermica-release.jks` — **fuera del control de versiones** | Firma del APK release | **Crítico**: perder el keystore hace imposible volver a firmar actualizaciones de esta app. Mantener una copia de respaldo cifrada fuera del repo; nunca en el repo ni en disco compartido |
| `HF_TOKEN` | `.env` (gitignored) + variable de entorno en Render | Lectura (read-only) de los repos remotos de modelos y métricas | Crear token nuevo con el mismo scope mínimo, actualizar `.env` y Render, verificar, revocar el viejo |
| `RENDER_API_KEY` | `.env` | API de Render (gestión del servicio desplegado) | Mismo patrón: nueva clave → actualizar → verificar → revocar |
| `GEMINI_API_KEY` / `OPENROUTER_API_KEY` | `.env` | Servicios auxiliares usados en análisis/herramientas | Mismo patrón de rotación |

Reglas:
- **Nunca loguear ni imprimir tokens** ni fragmentos de ellos (ni en consola, ni en tests, ni en la app). En la app, los errores de red muestran mensajes genéricos sin encabezados ni cuerpo de respuestas crudas.
- **Nunca** commitear `.env`, `local.properties`, ni `*.jks`/`*.keystore`. El `.gitignore` ya los excluye; si un secreto llegó a un commit, se considera comprometido: rotar de inmediato, no "borrar el commit".
- La comparación de claves en el servidor se hace en tiempo constante; la app nunca registra el valor de `APP_API_KEY`.

## Blindaje técnico del producto

- **HTTPS obligatorio** en todas las llamadas de la app; el manifiesto permite únicamente el permiso `INTERNET` necesario para la actualización del modelo.
- El modelo base va **empaquetado** en el APK y se **verifica por SHA-256** en el primer arranque antes de usarse.
- Los modelos descargados pasan **verificación de integridad (SHA-256 + tamaño) antes de reemplazar** el modelo instalado; la escritura es atómica (descarga a archivo temporal → rename).
- El historial local (Room) vive en almacenamiento interno; sin telemetría, sin analítica, sin terceros.
- El servidor expone solo cuatro rutas; `/model/latest` y `/model/download` exigen Bearer y comparación en tiempo constante; el servidor no escribe en los repos remotos.

## Reporte de vulnerabilidades

**No abras un issue público con detalles explotables.**

1. Canal preferido: un issue privado en este repositorio con la etiqueta **`security`** (o contacto directo con el dueño del repo a través de su perfil).
2. Incluí: palabra **confidencial** en el asunto, descripción, pasos/PoC, versión afectada (app/modelo/servidor) e impacto.
3. Recibirás acuse de lectura y un plan (parche o mitigación) dentro de un plazo razonable.
4. No divulgues hasta que exista fix publicado o mitigación acordada.

Reportá también **credenciales filtradas o sospechadas** (por ejemplo, un token pegado en un log público), aunque no sea una vulnerabilidad explotable al momento.

## Datos personales

La app no recolecta datos personales ni los envía a terceros. Las imágenes evaluadas permanecen en el dispositivo del usuario. El servidor solo entrega artefactos del modelo; no registra imágenes ni datos clínicos.
