# Changelog

Cronología resumida del proyecto (S0 → release 1.0.0). Fechas reales observadas en el trabajo del repositorio; el plan arrancó el **2026-08-03** y el release se cerró el **2026-11-17**.

| Fecha | Fase | Hito |
|-------|------|------|
| 2026-11-16 → 2026-11-17 | REFACTOR GRANDE (fases 0–8) | Reorganización en productos (`android/`, `pipeline/`, `model/`, `server/`), `uv` workspace único con lock pineado, `dermica-core` compartida, CI + Dependabot, Swagger pública, documentación de la API en español de los 43 `.kt` y ~40 `.py`; release 1.0.0 intacto; cero menciones a rutas fuera del repo, guardado por CI |
| 2026-08-03 | S0 | Documentación y definición de producto; arranque del plan (5 sprints × 2 semanas) |
| 2026-11-13 | — | Estructura inicial del monorepo (`BACKEND/` + `FRONTEND/`) |
| 2026-11-13/14 | S1 | Auditoría ISIC 2019: 25 331 imágenes de entrenamiento, 8 238 de test, BCC ≈ 13.1 % del train |
| 2026-11-14 | S2 | Subconjuntos TESIS (450 picks: 150 BCC / 300 no-BCC; 30 subconjuntos × 15; ≥ 5 BCC por subconjunto; semilla 42) y splits 70/20/10 estratificados agrupados por `lesion_id` (0 lesiones compartidas); manifiestos publicados en el dataset remoto |
| 2026-11-14 → 2026-11-16 | S3 | Modelado: evaluación de candidatos/variantes y entrenamiento MobileNetV3-Small; early stopping; mejor época 23; **AUPRC val 0.7788** |
| 2026-11-16 | S4 | Evaluación sobre TEST (una sola pasada, **AUPRC 0.7709**); exportación TFLite; **float16 elegido** (Δ AUPRC ≈ −0.0002; int8 destruía −0.331 y se descartó); FastAPI de distribución en la nube pública (`/healthz`, `/model/version`, `/model/latest`, `/model/download`) |
| 2026-11-16 → 2026-11-17 | S5 | App Android nativa (Kotlin + Compose Material 3 + LiteRT): inferencia on-device, Room (historial), modo subconjunto, actualización del modelo vía backend; ~20 commits, 2 PRs; APK release firmado (`APP_DERMICA-v1.apk`, ~22 MB, R8, firma v2) |
| 2026-11-16 | Release | Artefacto **float16 1.0.0** publicado como release completo |
| 2026-11-17 | Release | **tag `v1.0.0`** en el repositorio; APK release publicado (`APP_DERMICA-v1.apk`, 22.26 MB); sha256 `e5eb7077…`; métricas `1.0.0` publicadas |

## [1.0.0] — 2026-11-17

### Añadido
- App Android nativa: tamizaje on-device con el modelo base empaquetado (verificado por SHA-256 en el primer arranque), actualización opcional del modelo vía backend, historial local, modo subconjunto con ground truth por subconjunto.
- Servidor FastAPI público: `/healthz` y `/model/version` abiertos; `/model/latest` y `/model/download` con Bearer.
- Métricas de release: AUPRC 0.7788 (val) / 0.7709 (test); clasificación binaria BCC vs no-BCC.

### Sellado
- El artefacto float16 1.0.0 y sus métricas quedan **congelados**: solo un nuevo tag de versión puede cambiarlos (ver `ARCHITECTURE.md`).

## [Unreleased]

### Añadido
- Fase 0 (rama `refactor/00-contracts`): documentación fundacional — README recreado, `ARCHITECTURE.md` con contratos de artefactos, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `LICENSE` (MIT), `doc-adr/0001–0005`, plantillas de PR e issues.
- Fase 1 (pendiente): reorganización a `pipeline/`, `model/`, `server/`, `android/`.
