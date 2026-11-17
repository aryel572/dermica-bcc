# ADR 0004 — Monorepo con 4 productos y contratos de artefactos

- **Estado**: Aceptada (Fase 0; materializa en Fase 1)
- **Fecha**: 2026-11-17
- **Alcance**: repositorio completo

## Contexto

El proyecto nació como dos carpetas grandes (`BACKEND/` y `FRONTEND/`) que ya ocultan cuatro productos con ciclos de vida y herramientas distintos: pipeline de datos (Python), entrenamiento (Python/TF), servidor (FastAPI) y app (Kotlin/Android). Mezclarlos dificulta entender qué es qué, qué depende de qué, y dónde está el contrato entre entrenamiento y servidor o entre servidor y app.

## Decisión

El repositorio se reorganiza (Fase 1, sin cambiar lógica) en **4 productos raíz**:

```
pipeline/   model/   server/   android/
```

La comunicación entre productos es **únicamente** por artefactos-contrato (ver `ARCHITECTURE.md`): CSV/JSON versionados en el pipeline (dataset remoto), binario + `metrics.json` en el modelo (repos remotos de modelos/métricas), y contrato HTTP `/model/*` entre servidor y app. **Cero dependencias de código entre productos**: ningún producto importa a otro; cada uno se construye y testea de forma aislada.

## Consecuencias

- Mapa del repo explícito: un producto = una carpeta = una responsabilidad.
- El servidor no necesita el código del modelo ni del pipeline; la app no necesita ninguno de los otros tres.
- Los contratos de artefactos quedan documentados en `ARCHITECTURE.md` y son la única superficie compartida (cambio de contrato = PR explícito + ADR).
- La Fase 1 es un movimiento de carpetas: el riesgo se limita a rutas de build/CI, no a lógica.
- Empaquetar todo como un único módulo (o separar en 4 repos) se descarta: el primero mezcla herramientas, el segundo rompe la atomicidad de los cambios de contrato.
