## Alcance

<!-- Qué parte del repo toca este PR (fase/unidad) y qué decisiones de contrato implica -->

## Cambios

<!-- Lista breve: archivos afectados y qué cambia en cada uno -->

## Estrategia de verificación

<!-- Comandos ejecutados y salida relevante (tests, build, compuertas --verificar). Build fresco. -->

- [ ] `./gradlew :app:testDebugUnitTest` en verde (si toca `android/`)
- [ ] `./gradlew :app:assembleDebug` compila (si toca `android/`)
- [ ] Compuertas `--verificar` / tests de pipeline o modelo en verde (si toca `pipeline/` o `model/`)
- [ ] Sin secretos, sin binarios, sin rutas fuera del repo

## Riesgo

<!-- Una línea: qué puede romperse y cuál es el plan de rollback/mitigación -->
