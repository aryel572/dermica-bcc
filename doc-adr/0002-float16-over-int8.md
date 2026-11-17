# ADR 0002 — float16 sobre int8 para el release

- **Estado**: Aceptada (decisión tomada en S4; registrada en Fase 0)
- **Fecha**: 2026-11-17
- **Alcance**: `model/` (hoy `BACKEND/model/`), distribución vía `server/`

## Contexto

El modelo debe correr on-device en Android vía LiteRT. La cuantización reduce el tamaño y el costo de inferencia, pero int8 y float16 no preservan la calidad por igual para este MobileNetV3-Small fine-tuned sobre BCC desbalanceado (AUPRC como métrica primaria).

Resultados medidos (misma evaluación, umbral 0.5):
- **int8**: Δ AUPRC ≈ **−0.331** — destrucción de calidad; inutilizable para tamizaje.
- **float16**: Δ AUPRC ≈ **−0.0002** — indistinguible del float32 en la práctica.

## Decisión

Se publica **float16** como único artefacto del release **1.0.0** (`model-float16.tflite`), empaquetado en la app y disponible en el servidor. El candidato int8 se descarta definitivamente para este modelo y dataset.

## Consecuencias

- Artefacto 1.0.0 ≈ 1.9 MB; calidad preservada (AUPRC val 0.7788 / test 0.7709).
- La cuantización queda registrada en `dermica-metrics/1.0.0.json` (`quantization: float16`, con `float16_artifact`, `float16_size_bytes`, `float16_sha256`).
- 1.0.0 queda sellado: su binario y sus métricas no se modifican; cualquier mejora exige nuevo tag `X.Y.Z`.
- Dispositivos con LiteRT y soporte FP16 (todos los Android objetivo del proyecto) lo ejecutan nativamente; no hay fallback int8.
- Si en el futuro se retoma int8, será un experimento con evaluación completa y decision propia — nunca un "parche" al release.
