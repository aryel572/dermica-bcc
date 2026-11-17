---
license: cc-by-nc-4.0
task_categories:
- image-classification
tags:
- isic2019
- dermatology
pretty_name: ISIC 2019 — manifiestos de entrenamiento
---

# ISIC 2019 — Manifiestos de entrenamiento

Manifiestos y configuración derivados de **ISIC 2019** para el entrenamiento del
modelo de tamizaje de carcinoma basocelular (BCC). **Este repositorio nunca
almacena imágenes**; contiene únicamente texto (CSV/JSON/Markdown). Las
imágenes ISIC se obtienen del origen oficial con su licencia.

Contenido en español; idioma del dataset: inglés (metadatos y etiquetas ISIC).

## Contenido del repo git-remote

| Archivo | Contenido |
|---------|-----------|
| `splits` → `train.csv` | Split 70%: `image, split, label_full, label_bcc, lesion_id` |
| `splits` → `val.csv`   | Split 20% (mismas columnas) |
| `splits` → `test.csv`  | Split 10% (mismas columnas) |
| carpeta `tesis` → `tesis_manifest.csv` | 450 picks (30 × 15) con `sha256` de cada copia |
| carpeta `tesis` → `tesis_exclude.csv`  | 447 `image_id` únicos a excluir del pipeline |
| `config.json`      | Semillas, ratios y parámetros (semilla 42) |

## Forma de los datos

- Fuente: **ISIC 2019** (CSV oficiales de la versión 2019). El test oficial de
  ISIC **no entra en ningún split** (medición final únicamente).
- **25.331 imágenes de entrenamiento** originales (8.238 de test oficial).
- Split 70/20/10 **agrupado por `lesion_id`** (StratifiedGroupKFold,
  `n_splits=10`, `shuffle=True`, `random_state=42`; reunión 7 folds → train,
  2 → val, 1 → test) y **estratificado por BCC/no-BCC**. Los sin
  `lesion_id` se tratan como singletons. **0 lesiones compartidas** entre
  particiones.
- **TESIS excluida** mediante `tesis_exclude.csv` (carpeta `tesis`) antes de
  particionar: **450 imágenes** elegidas en 30 subconjuntos × 15, con
  repetición permitida entre subconjuntos, reservadas como demo de evidencia
  y nunca entrenadas.
- Conteos reales: 25.331 train originales → **24.884** candidatos
  (17.419 / 4.977 / 2.488; BCC 12,76 % / 12,78 % / 12,74 %).
- Manifiesto (`tesis_manifest.csv`): una fila por pick con columnas
  `subset_id, image_id, label_full, sha256, original_path`. Los CSV de
  partición llevan `image, split, label_full, label_bcc, lesion_id`.
- Reproducibilidad: semilla 42 en `config.json`; manifiestos con hash por imagen.

## Atribución y licencia

Datos: **ISIC 2019** — International Skin Imaging Collaboration
(HAM10000 — Tschandl et al.; BCN20000 — Hospital Clínic de Barcelona;
MSK — Memorial Sloan Kettering Cancer Center), licencia
**CC BY-NC 4.0**. Uso no comercial con atribución. No se redistribuyen
imágenes; estos manifiestos son solo índices y metadatos derivados.

## Notas

- Sin imágenes ni binarios ISIC, sin claves ni credenciales.
- `lesion_id` ausente se representa como cadena vacía.
