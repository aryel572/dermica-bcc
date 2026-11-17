# ADR 0005 — Cero rutas fuera del repositorio

- **Estado**: Aceptada (Fase 0)
- **Fecha**: 2026-11-17
- **Alcance**: todo el contenido del repositorio

## Contexto

Un clon fresco del repositorio debe ser auto-contenido: quien lo abre (revisores, CI, futuros mantenedores) no puede depender de la máquina de quien lo escribió. Las referencias a rutas locales absolutas del disco del autor hacen la documentación no auditable, los scripts no portables y los PRs imposibles de revisar desde otro entorno.

## Decisión

**Ningún archivo del repositorio menciona rutas o ubicaciones fuera del repositorio**: ni en docs, ni en comentarios, ni en scripts, ni en ejemplos, ni como "aclaración". Se permiten:

- **Rutas relativas al repo** (`BACKEND/...`, `android/...`, `./gradlew`), que funcionan en cualquier clon.
- **Nombres de repositorios remotos** (el dataset, los modelos y las métricas remotos), invocados por nombre como parte del contrato de distribución.

No se permiten rutas absolutas del sistema, carpetas personales, discos externos ni utilidades específicas de la máquina del autor. Los datos crudos se ubican donde el entorno del ejecutor los provea, y los scripts los reciben por parámetro — el artefacto commiteado nunca los referencia por ruta.

## Consecuencias

- La documentación y los ejemplos son reproducibles desde cualquier clon, sin el entorno original.
- La revisión de PRs no depende de conocer el disco de nadie.
- Las ejecuciones de pipeline/modelo reciben ubicación de datos por CLI/config, no por constante commiteada.
- El coste es disciplina: cada ejemplo o ruta nueva se revisa contra esta regla antes de mergear (verificable con búsqueda de patrones prohibidos).
- Conexión de fase: la CI de fases posteriores puede rechazar contenido con rutas externas como guardia mecánica.
