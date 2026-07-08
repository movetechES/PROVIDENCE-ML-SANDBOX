# CLAUDE.md — PROVIDENCE-ML-SANDBOX

## Propósito y frontera de licencias (norma)

Este repo es el sandbox de evaluación de detectores del proyecto PROVIDENCE
(github.com/movetechES/PROVIDENCE). Existe por una razón de licencias: aquí se evalúan
comparativamente candidatos AGPL (Ultralytics YOLO) y permisivos (RT-DETRv2, D-FINE,
Apache-2.0); al repo del producto SOLO cruzan los pesos ONNX del ganador, el informe
del benchmark y la decisión de licencia registrada. Nada AGPL toca el producto ni su
pipeline de datos (incluida la pre-anotación, que se hace siempre con el candidato
permisivo).

## Reglas de trabajo heredadas del producto

- Toda dependencia se propone con versión y licencia ANTES de tocar cualquier
  manifiesto (requirements.txt incluido).
- Las descargas e instalaciones (paquetes, pesos, datasets) las ejecuta Alejandro;
  Claude las propone, no las ejecuta.
- Evidencia = salida real de comandos. No se afirma que algo funciona sin haberlo
  ejecutado y mostrado la salida.
- Credenciales jamás en el repo: ni en código, ni en configs, ni en historial.
- El límite de efectores del producto aplica igual aquí.

## Reglas propias del sandbox

a) Los DATOS viven en `C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA\` (versionados con DVC),
   nunca en este repo. Los pesos descargados y los checkpoints tampoco se commitean
   (ver `.gitignore`).

b) Todo dataset requiere revisión de licencia previa por Alejandro y datasheet. Lo
   NC (no comercial) queda excluido de cualquier papel, incluida la evaluación.

c) El split honesto es intocable: la asignación de sesiones a train/validación/test
   está fijada en el manifiesto del split (copia en este repo, original en
   docs/datos/ del producto). El test ciego (casa) se evalúa UNA vez.

d) Todo experimento queda registrado en MLflow local con semilla, config y versión
   DVC de los datos (reproducibilidad, plan_ml §0.6).

e) El criterio de aceptación y la regla de decisión del benchmark están fijados en
   DECISIONES.md §15 del producto y NO se ajustan desde aquí.

## Estructura

- `harness/` — evaluación: métricas, splits, latencia.
- `configs/` — configs de entrenamiento/eval versionadas.
- `informes/` — resultados del benchmark.
- `scripts/` — utilidades.
- `REGISTRO.md` — registro de decisiones propio del sandbox (marco normativo:
  DECISIONES.md §13–§17 del producto).

No hay carpeta de datos: los datos no viven aquí (regla a).
