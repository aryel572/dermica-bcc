# ADR 0001 — Split agrupado por `lesion_id`

- **Estado**: Aceptada (decisión tomada en S2; registrada en Fase 0)
- **Fecha**: 2026-11-17
- **Alcance**: `pipeline/` (hoy `BACKEND/data_pipeline/splits/`)

## Contexto

ISIC 2019 contiene múltiples imágenes por lesión (mismo `lesion_id`, distinto punto de vista/momento/iluminación). Un split aleatorio por imagen filtra la misma lesión a train y test: las métricas quedan infladas y no representan al paciente nuevo (la única unidad clínicamente relevante para el tamizaje).

Además, el TEST oficial de ISIC no trae `lesion_id` confiable (24.9 % `UNK`), por lo que todo split se construye **exclusivamente sobre el TRAIN** y el TEST oficial queda sellado como evaluación de una sola pasada.

## Decisión

Split 70/20/10 **agrupado por `lesion_id`** (una lesión completa cae en una sola partición), **estratificado por clase BCC/no-BCC**, con **semilla fija 42**, tras **excluir los 447 image_id únicos de TESIS** (`tesis_exclude.csv`). Las entradas sin `lesion_id` se agrupan juntas como un grupo propio. Compuertas inline: ningún artefacto se escribe si alguna falla (duplicados, ratios desviados > 1 punto porcentual, etc.).

## Consecuencias

- Cero lesiones compartidas entre particiones: las métricas de validación/test son representativas del despliegue.
- Los ratios quedan ~70/20/10 con desviaciones mínimas por el agrupamiento (tolerancia ±1 p.p.; se registra en el reporte).
- El conteo de BCC por partición es comparable entre train/val/test (estratificación).
- El TEST oficial se usa una única vez (S4); no entra en ningún split ni en la iteración del modelo.
- Cualquier cambio al criterio de agrupamiento invalida métricas y manifiestos: exige ADR nuevo y rehacer los splits.
