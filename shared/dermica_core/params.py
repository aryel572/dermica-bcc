"""Parámetros congelados del release 1.0.0 y del monorepo.

Qué contiene: `EXPECTED_SHA256`, el SHA-256 del asset `model-float16.tflite`
que la app Android empaqueta y verifica antes de usarse.

Constituye el contrato textual del release para cualquier recomputación en
Python (p. ej. auditar que el asset del repo coincide con el valor publicado).
Fase del pipeline: release 1.0.0 (S4) y empaquetado de la app Android (S5); el
certificado vive también en ADR-0003 y en el test de release de la app.

Qué recibe y qué produce: no lee flags ni escribe artefactos; expone valores
constantes importables. Consumidores directos en Python: ninguno hoy (las
verificaciones de la app Android viven en Kotlin); existe para que la tesis y
futuras auditorías Python no dupliquen el literal.

Procedencia de EXPECTED_SHA256
------------------------------
Release 1.0.0 (tag `v1.0.0`): el modelo base va empaquetado como asset de la
app (`android/app/src/main/assets/model-float16.tflite`, 1 945 356 bytes) y la
app lo verifica por SHA-256 en el primer arranque. El valor está fijado en dos
lugares del árbol Android — `android/app/src/main/java/com/dermica/bcc/data/model/ModelManager.kt`
(`EXPECTED_SHA256`) y `android/app/src/test/java/com/dermica/bcc/ModelReleaseTest.kt`
— y registrado en el ADR-0003 (`doc-adr/0003-app-base-packaged-model.md`) y en
`android/README.md`. Aquí se declara como constante del contrato del release
para que cualquier recomputación en Python (p. ej. auditar que el asset del
repo coincide con el valor publicado) tenga una única fuente textual.

Qué NO vive aquí (decisión medida, no olvido): la semilla (42), los tamaños de
TESIS (30 subconjuntos × 15 imágenes, mínimo 5 BCC), los ratios del split
70/20/10, el umbral 0.5 y el ΔAUPRC 0.02 NO se duplican en Python; su fuente de
verdad en runtime es `pipeline/config.json`, contrato congelado citado por la
tesis. Ningún módulo Python los escribe como literales; meterlos aquí los
dejaría sin consumidor y mentiría como "única fuente".
"""

# SHA-256 del asset model-float16.tflite del release 1.0.0 (1 945 356 bytes).
# Procede de ADR-0003 / ModelManager.kt EXPECTED_SHA256 (ver docstring del módulo).
# Consumidor esperado: cualquier auditoría Python del asset del release y la
# sincronía con el valor que la app Android verifica en su primer arranque.
EXPECTED_SHA256: str = "d62c401c5f66a7eccd5f752b2d7aa73964fe6f75a92acde3cae951b0d911cc0a"
