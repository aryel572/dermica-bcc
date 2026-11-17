# CONTRIBUTING — APP_DERMICA

Gracias por contribuir. Estas convenciones son obligatorias para todo el historial del repo.

## Ramas y flujo de cambios

- Trabajo nuevo = rama desde `main` actualizado: `feat/<unidad>`, `docs/<unidad>` o `refactor/<fase>`.
- **Un merge por fase/unidad, vía Pull Request**; el dueño del repo hace los merges, nunca quien envía el PR.
- Commits **atómicos y pequeños**; cada commit compila/testea si toca código ejecutable.
- PRs enlazan su evidencia (comandos + salida, resumen de qué verificó).

## Commits

- **Conventional Commits**: tipo en inglés + descripción en español.
  - ✅ `feat(pipeline): add TESIS subset formation`
  - ✅ `fix(model): corregir preprocess en evaluate`
  - ❌ `cambios varios`, `fix stuff`, `update files`
- **Sin atribución de IA** ni `Co-Authored-By`: ningún trailer menciona herramientas ni modelos.
- Tipos frecuentes: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

## Reglas de tests y build

- **Unit tests verdes obligatorios** antes de cada push que toque código (`./gradlew :app:testDebugUnitTest` en la app; `--verificar` donde exista ese modo).
- Android: `assembleDebug` compila **y** `testDebugUnitTest` queda en verde **antes de pedir review**. El APK release solo se genera para release y con firma local.
- **Nada se mergea sin un build fresco en verde** (checkout actualizado, no artefactos viejos).
- Los tests no dependen de servicios externos ni de credenciales (se usan fakes/mocks o datos de prueba locales).

## Estilo de código

- Identificadores y nombres de archivo: inglés.
- **Comentarios y docstrings: siempre en español** (KDoc, docstrings Python), aunque el identificador sea inglés. Documentan el *porqué* de lo no obvio; las constantes mágicas llevan comentario.
- Código mínimo: reutilizá lo que ya existe en el producto antes de crear abstracciones; nada de opciones "por si acaso".
- Un archivo = una responsabilidad visible; funciones pequeñas; early returns.
- Sin dependencias nuevas sin justificación en el PR.

## Secretos y contenido prohibido en el repo

- **Nunca** commitear: `.env`, `local.properties`, keystores (`*.jks`, `*.keystore`), tokens, imágenes ISIC, binarios grandes ni salidas de corridas.
- Los datos ISIC se citan y licencian según `LICENSE`; el repo solo aloja manifiestos/texto con atribución.
- **Cero rutas fuera del repositorio** en código, comentarios, docs o scripts: todo es auto-contenido y relativo (ver `doc-adr/0005-zero-external-paths.md`).
- Hugging Face se invoca por **nombre de repositorio remoto** (p. ej. `dermica-models`), nunca por rutas locales ni por credenciales inlineada.

## Alcance de los cambios

- Fase 0 (contratos): solo documentación en raíz y `.github/`. Fase 1 mueve carpetas. La CI vive únicamente en `.github/workflows/` y solo en su fase.
- No se tocan `render.yaml`, ni los workflows, ni los contratos de `ARCHITECTURE.md` fuera de su proceso (PR + actualización de doc + ADR si aplica).
- Cambiar un contrato de artefacto (esquema UDL, `metrics.json`, `ground_truth.json`, HTTP) exige actualizar `ARCHITECTURE.md` y, si cambia una decisión, un ADR.

## Proceso de PR

1. Abrí el PR con la plantilla (`Alcance` / `Cambios` / `Estrategia de verificación` / `Riesgo`).
2. Adjuntá comandos y salida real de verificación (tests, build).
3. El reviewer verifica contra `main` fresco; los comentarios se resuelven en el mismo PR.
4. Merge con squash o merge commit según el historial del repo; nunca push directo a `main`.

## Estilo de revisión

- Señalá el problema y el porqué técnico; proponé el cambio concreto.
- "Pedí menos, no más": si una abstracción no simplifica al consumidor actual, no entra.
