# Architecture Decision Records (ADR)

Decisiones de arquitectura del proyecto APP_DERMICA, una por archivo. Cada ADR registra **contexto → decisión → consecuencias**; una vez aceptada, la decisión solo cambia con un ADR nuevo que la reemplace.

| # | Título | Estado | Fecha |
|---|--------|--------|-------|
| [0001](0001-split-group-by-lesion-id.md) | Split agrupado por `lesion_id` | Aceptada | 2026-11-17 (decisión tomada en S2) |
| [0002](0002-float16-over-int8.md) | float16 sobre int8 para el release | Aceptada | 2026-11-17 (decisión tomada en S4) |
| [0003](0003-app-base-packaged-model.md) | Modelo base empaquetado en la app | Aceptada | 2026-11-17 (decisión tomada en S5) |
| [0004](0004-repo-split-into-products.md) | Monorepo con 4 productos y contratos de artefactos | Aceptada | 2026-11-17 (Fase 0) |
| [0005](0005-zero-external-paths.md) | Cero rutas fuera del repositorio | Aceptada | 2026-11-17 (Fase 0) |

Formato: contexto, decisión, consecuencias (10–30 líneas). Sin meta-narración; lenguaje neutro y español.
