# REGISTRO.md — Registro de decisiones del sandbox

Marco normativo: **DECISIONES.md §13–§17 del repo del producto (PROVIDENCE)**. Este
registro documenta decisiones operativas propias del sandbox y no puede contradecir
ni ajustar lo fijado allí; en particular, el criterio de aceptación y la regla de
decisión del benchmark (§15) se leen del producto, no de aquí.

## Dependencias aprobadas por Alejandro (2026-07-08)

Propuestas con versión y licencia antes de tocar manifiesto; instalaciones y
descargas las ejecuta Alejandro en el venv propio del sandbox.

| Dependencia | Serie aprobada | Licencia | Para qué |
|---|---|---|---|
| Python | 3.12 | PSF | Intérprete del venv del sandbox |
| torch + torchvision | >= 2.7, build CUDA 12.8 (cu128) | BSD-3-Clause | Framework. Restricción dura: la RTX 5070 es Blackwell (sm_120); builds anteriores a 2.7/cu128 no la soportan — fija el suelo de versiones del resto |
| ultralytics | 8.3.x | AGPL-3.0 | Candidato AGPL (YOLO). Solo sandbox; jamás cruza al producto ni pre-anota |
| transformers | 4.x reciente (exacta al resolver) | Apache-2.0 | Vehículo de RT-DETRv2 y D-FINE (pesos Apache-2.0): un solo harness para ambos candidatos permisivos |
| pycocotools | 2.x | BSD | mAP formato COCO — métrica del benchmark, idéntica para todos |
| onnx / onnxruntime-gpu | actuales | Apache-2.0 / MIT | Exportación del ganador y medida de latencia |
| opencv-python | 4.x | Apache-2.0 | Decodificación de vídeo y visualización de errores |
| dvc | 3.x | Apache-2.0 | Versionado de PROVIDENCE-DATA (plan_ml §2.5) |
| mlflow | 2.x/3.x | Apache-2.0 | Registro de experimentos LOCAL — elegido sobre W&B para que métricas, rutas y nombres de sesión no salgan a un SaaS de terceros |
| numpy | 2.x | BSD | Directa por SBOM |

Anotaciones normativas de la aprobación:

1. **La latencia del benchmark se mide en onnxruntime-gpu PORQUE ONNX es el formato
   que cruza al producto**: es el proxy honesto de la latencia que importa (la del
   artefacto entregable), no la del runtime de entrenamiento de cada framework. Los
   tres candidatos se exportan a ONNX y se miden en las mismas condiciones (FP16,
   640 px, RTX 5070). Así se registra y así se reporta en el informe del benchmark.

2. **Postura sobre ffmpeg (registrada ya, no al llegar a B5)**: ffmpeg se usa como
   proceso externo invocado por tools/capture del repo del producto — no enlazado,
   no distribuido con el producto. Su licencia (LGPL/GPL según build) no alcanza al
   producto bajo esa postura. Cualquier cambio de uso (enlazado, redistribución)
   reabre la decisión.

Aviso registrado para B5 (aprobación formal cuando llegue): CVAT (MIT, servicio
Docker local) como herramienta de anotación.
