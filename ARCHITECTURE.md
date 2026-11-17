# ARCHITECTURE — APP_DERMICA

Arquitectura de los 4 productos del monorepo y **contratos de artefactos** entre ellos. Los contratos son archivos (CSV/JSON) y un contrato HTTP; nunca código compartido.

## Flujo general

```
ISIC 2019 (origen externo; solo lectura, jamás se redistribuye)
        │
        ▼
┌───────────┐   train/val/test.csv    ┌───────────┐   model-float16.tflite   ┌───────────┐
│ PIPELINE  │ ──────────────────────▶ │   MODEL   │ ───────────────────────▶ │  SERVER   │
│           │   tesis_exclude.csv     │           │   dermica-metrics/       │  (FastAPI)│
└───────────┘                         │           │      <tag>.json          │           │
        │                             └───────────┘                          └───────────┘
        │ ground_truth.json (30 subconjuntos)                                      ▲
        ▼                                                                            │ HTTPS
┌───────────────────────────────────────────────────────────────────────────────────────┐
│ ANDROID (app Kotlin + Compose + LiteRT)                                              │
│  • modelo base empaquetado (verificado por SHA-256 en el primer arranque)           │
│  • actualización opcional: /model/version → /model/latest → /model/download           │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Sin flechas de vuelta: la app nunca envía imágenes ni telemetría al servidor; el servidor nunca escribe en el dataset remoto; el pipeline nunca escribe sobre los datos crudos.

## Productos y ubicación

| Producto | Ubicación | Responsabilidad única |
|----------|-----------|------------------------|
| pipeline | `pipeline/` | Auditar ISIC, formar TESIS, splits 70/20/10, manifiestos |
| model    | `model/` | Entrenar/evaluar MobileNetV3-Small, exportar TFLite, publicar métricas |
| server   | `server/` | Distribuir el release por HTTP (Render, free) |
| android  | `android/` | Tamizaje on-device + historial local + actualización del modelo |

## Artefactos-contrato (4 + 1)

| # | Artefacto | Productor | Consumidor | Esquema |
|---|-----------|-----------|------------|---------|
| 1 | `train.csv` · `val.csv` · `test.csv` | pipeline | model | **Esquema UDL** (columnas fijas, orden contractual): `image` (image_id ISIC), `split` (`train`, `val` o `test`), `label_full` (diagnóstico ISIC completo), `label_bcc` (0/1), `lesion_id` (cadena; vacío si ausente, nunca `NaN`). CSV UTF-8 con cabecera; `label_bcc` es 0/1 estricto; partición agrupada por `lesion_id` (cero lesiones compartidas) y estratificada por BCC. |
| 2 | `tesis_exclude.csv` | pipeline | pipeline · model | Una columna `image_id`; 447 IDs únicos a excluir del split (procedentes de 450 picks TESIS: 150 BCC / 300 no-BCC, 30 subconjuntos × 15 imágenes, ≥ 5 BCC por subconjunto, repeticiones permitidas, seed 42). Los IDs excluidos nunca aparecen en `train/val/test.csv`. |
| 3 | `dermica-metrics/<tag>.json` (release actual: `1.0.0.json`) | model · server (solo lectura) | server · android | JSON UTF-8 con: `tag` (str, `X.Y.Z`), `auprc` (float, métrica primaria), `recall` (float), `sensibilidad` (float, alias clínico de `recall`), `especificidad` (float), `balanced_acc` (float), `f2` (float), `precision` (float), `tp` · `fp` · `tn` · `fn` (int), `quantization` (str, `float16`), `float16_artifact` (str, nombre del `.tflite` dentro del repo remoto del modelo), `float16_size_bytes` (int), `float16_sha256` (str hex SHA-256). |
| 4 | `ground_truth.json` | pipeline | android | Objeto JSON plano `{filename: etiqueta_ISIC}` (p. ej. `"ISIC_0026350.jpg": "BCC"`); un archivo por subconjunto de demostración (30); la etiqueta es el literal ISIC exacto y la app hace comparación exacta. Sin campos extra de metadatos. |
| +1 | `tesis_manifest.csv` | pipeline | model · auditoría | 450 filas (una por pick, repeticiones incluidas) con, como mínimo, el `image_id` y el `sha256` de cada copia; sirve para auditar la formación de TESIS y verificar artefactos reproducidos. |

Notas de los contratos:
- Los CSV de split siempre se escriben completos o ninguno (si una compuerta del pipeline falla, no se emite nada).
- `lesion_id` se escribe como cadena vacía cuando el dato es ausente; los consumidores nunca verán `null`/`NaN`.
- Las métricas son documentales: nunca tiran abajo un entrenamiento (divisiones por cero ⇒ `0.0`).

## Reglas de dependencia (no negociables)

1. **pipeline → nada.** No importa código de `model`, `server` ni `android`. Lee datos crudos (solo lectura) y produce CSVs/JSON/manifiestos.
2. **model → nada.** No importa pipeline, server ni android. Consume los CSVs del contrato #1 y produce el artefacto TFLite + `metrics.json`.
3. **server → nada fuera de los repos remotos de modelos/métricas.** No importa pipeline ni model ni android; solo HF remoto (lectura) y FastAPI. Sin escritura sobre los repos remotos.
4. **android → solo el contrato HTTP** con el servidor (`/model/version` público, `/model/latest` y `/model/download` con Bearer) y el asset empaquetado. Sin imports de otros productos, sin almacenamiento de secretos propios, sin telemetría.
5. **Cero dependencias de código entre productos.** El pegamento son archivos y HTTP; un producto se construye y testea sin los demás.
6. **Las rutas son relativas al repo** (ver `doc-adr/0005-zero-external-paths.md`); el servidor apunta a repos remotos por nombre, nunca a rutas locales.

## Versionado y ciclo de vida de artefactos

- Versiones completas: **`X.Y.Z`** (hoy: **1.0.0**, publicación float16). Épocas de entrenamiento: **`X.Y.Z-epN`**, se van superando y borrando conforme avanza el entrenamiento.
- Al cerrar `X.Y.Z` solo queda la versión completa (binario + `X.Y.Z.json` de métricas); las `-epN` y sus métricas se borran.
- **Release 1.0.0 sellado**: `1.0.0.json` y el artefacto float16 **no se modifican jamás**. Cualquier cambio de métricas, binario o esquema exige un tag nuevo (`X.Y.Z+1` vía siguiente release) y un JSON nuevo.
- Los checkpoints de época viven en un repo remoto temporal y se borran al llegar al release.
- El dataset de manifiestos y los repos de modelos/métricas son remotos y privados; los nombres de repo remoto son parte del contrato, las credenciales nunca lo son (ver `SECURITY.md`).

## Qué se puede añadir y qué no

**Sí se puede (con PR y contrato):**
- Nuevas claves *opcionales* en JSON con defaults seguros y documentación actualizada.
- Nuevos productos nuevos (nunca como dependencia de uno existente) siguiendo el contrato de cero imports cruzados.
- Endpoints nuevos del servidor bajo una ruta nueva de versión (no romper `/model/*` 1.0.0).
- Nuevos manifiestos derivados (solo texto) con atribución ISIC CC BY-NC 4.0.

**No se puede:**
- Actualizar `dermica-metrics/1.0.0.json` ni el binario float16 de 1.0.0 (sellado).
- Relajar el esquema UDL (reordenar columnas o cambiar tipos rompe a todos los consumidores).
- Introducir dependencias de código entre productos (ni por "utilidad compartida").
- Commitear datos ISIC, binarios o credenciales.
- Romper el contrato HTTP de la app (`/model/*` tal como lo consume 1.0.0).

## Decisiones registradas

Las decisiones de arquitectura viven en `doc-adr/` (0001 split grupado · 0002 float16 · 0003 modelo base empaquetado · 0004 monorepo por productos · 0005 cero rutas externas).
