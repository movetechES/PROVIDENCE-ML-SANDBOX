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

### Resolución (2026-07-08)

Primera instalación verificada por Alejandro en el venv propio del sandbox.
Versiones resueltas tomadas de `requirements-lock.txt` (salida de `pip freeze`).

| Dependencia | Serie aprobada | Versión resuelta | ¿Casa con la serie? |
|---|---|---|---|
| torch | >= 2.7, cu128 | 2.11.0+cu128 | Sí |
| torchvision | >= 2.7 (pareja de torch), cu128 | 0.26.0+cu128 | Sí (pareja de torch 2.11) |
| ultralytics | 8.3.x | 8.3.253 | Sí |
| transformers | 4.x reciente (exacta al resolver) | 4.57.6 | Sí — esta es la exacta |
| pycocotools | 2.x | 2.0.11 | Sí |
| onnx | actuales | 1.22.0 | Sí |
| onnxruntime-gpu | actuales | 1.27.0 | Sí |
| opencv-python | 4.x | 4.13.0.92 | Sí |
| dvc | 3.x | 3.67.1 | Sí |
| mlflow | 2.x/3.x | 3.14.0 | Sí |
| numpy | 2.x | 2.4.4 | Sí |

Ninguna resolución fuera de su serie aprobada.

**Gate de instalación (verificado por Alejandro, 2026-07-08):**

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
2.11.0+cu128 True NVIDIA GeForce RTX 5070 Laptop GPU
```

Build cu128 confirmada, CUDA disponible y RTX 5070 detectada: la restricción dura
de Blackwell (sm_120) que fijaba el suelo de versiones queda satisfecha.

**Reproducibilidad:** `requirements-lock.txt` queda en el repo como eslabón local de
reproducibilidad del venv — mismo criterio que los locks de Fase 0 (DECISIONES.md §6
del producto). `requirements.txt` expresa las series aprobadas; el lock, la
resolución concreta verificada en esta fecha.
