# dermica-core — núcleo compartido de APP_DERMICA

Paquete único con los símbolos que usan dos o más productos del monorepo
(pipeline, model, server, android). Nada más: si un símbolo no tiene al menos
dos consumidores, no vive aquí.

## Qué contiene y por qué

| Módulo     | Contenido                                        | Consumidores (medidos)                                  |
|------------|--------------------------------------------------|---------------------------------------------------------|
| `labels`   | `ISIC_LABELS`, `BCC_LABEL`, `is_bcc()`           | `pipeline/splits`, `pipeline/tesis`, `pipeline/audit`   |
| `hashing`  | `sha256_file()`, `sha256_bytes()`                | `pipeline/audit`, `pipeline/tesis`, `pipeline/prep/export_float16` |
| `params`   | `EXPECTED_SHA256` (release 1.0.0)                | contrato con la app Android (ver docstring del módulo)  |
| `io`       | `leer_csv_csv_splits()` con validación           | `pipeline/prep` (resize_224, resize_224_multi, resize_224_test) |
| `cli`      | `ParserEspanol` (ArgumentParser en español)      | los 10 módulos CLI de `pipeline/` (antes copias idénticas) |

`io.py` importa pandas de forma **perezosa** dentro de la función: el core
sigue siendo instalable con cero dependencias externas.

Los parámetros de la corrida de datos (semilla, tamaños de TESIS, ratios de
split, umbral) **no** se duplican aquí: su fuente de verdad en runtime es
`pipeline/config.json`, contrato congelado citado por la tesis.
