# Línea base zero-shot — benchmark de detectores (paso B3, fase 2)

Fecha: 2026-07-16. Ejecutado en el sandbox (venv propio, RTX 5070 Laptop, enchufada
a corriente y en perfil de máximo rendimiento, verificado por Alejandro).

## 1. Contexto y mandato

Evaluación comparativa zero-shot (pesos COCO tal cual, sin fine-tuning) de los tres
candidatos fijados por DECISIONES.md §15 del producto: YOLO11 (ultralytics, AGPL-3.0,
solo sandbox) frente a RT-DETRv2 y D-FINE (Apache-2.0, vía transformers). El criterio
de aceptación se RATIFICA con esta línea base en mano; la decisión no se toma aquí.

**Estado**: COMPLETO. 12 corridas de mAP + 6 de latencia onnxruntime-gpu (con
sanity check FP16), 18 corridas en MLflow local. La latencia exigió sustituir la
build de onnxruntime-gpu (cu13→cu12, §6; registrado en REGISTRO.md y ejecutado
por Alejandro).

## 2. Protocolo ejecutado

### Pesos (descargados por Alejandro, 2026-07-16)

| Modelo | Peso | Fuente | Licencia | sha256 |
|---|---|---|---|---|
| yolo11s | yolo11s.pt | github.com/ultralytics/assets v8.3.0 | AGPL-3.0 | `85a76fe8…d502d5` |
| yolo11m | yolo11m.pt | ídem | AGPL-3.0 | `d5ffc1a6…305b95` |
| rtdetrv2-s | PekingU/rtdetr_v2_r18vd | HF Hub, snapshot `5650961749fa93567c0d46fc7f43ea4f9e914107` | Apache-2.0 | `d18309d0…c7115` (safetensors) |
| rtdetrv2-m | PekingU/rtdetr_v2_r34vd | HF Hub, snapshot `5d60d8fb5eb2a3ee5575bf200ecbce4e2cfa420c` | Apache-2.0 | `d67528a6…2d612` |
| dfine-s | ustc-community/dfine-small-coco | HF Hub, snapshot `f79e65b5fbb33ceb9d3ebba042955d7410c608f8` | Apache-2.0 | `a144baff…cd18f` |
| dfine-m | ustc-community/dfine-medium-coco | HF Hub, snapshot `4ab9f5e466432f4fcf9a9a023da6aa5ecc9c9829` | Apache-2.0 | `5a53154f…02634` |

sha256 completos y snapshots registrados en MLflow por corrida. Las variantes
`-obj2coco` de D-FINE NO se evaluaron: sus pesos no están descargados y la condición
de la aprobación era «solo si no cuesta nada»; quedan como corrida opcional futura.

### Datos y muestreo

- Versión DVC de los datos (md5 .dvc): virat `aec8d2c7…212c.dir`, meva
  `58a67ccd…86d7.dir`, meva-anotaciones `ccc73b41…26b2.dir`. Commit del sandbox
  en la ejecución: `fe7b59b`.
- Conversión a COCO según los mapeos aprobados (VIRAT 1→person, 2/3→vehicle,
  0/4/5 descartados; MEVA person/vehicle, other/bag descartados; predicciones
  por nombre: person→person, car/bus/truck→vehicle, bicycle/motorcycle
  descartadas).
- Muestreo determinista aprobado: frames con ≥1 GT ∩ rango decodificable;
  VIRAT zancada 90 frames / tope 20 por vídeo; MEVA zancada 150 / tope 25 por clip.
- Resultado: **VIRAT 4.632 frames de 315 vídeos** (61.979 cajas GT: 46.943 vehicle,
  15.036 person); **MEVA 2.170 frames de 170 clips** (4.294 cajas GT: 3.780 person,
  514 vehicle; 41 clips «marked empty» quedaron fuera por construcción).
  0 frames fallidos en la extracción (JPEG q95; 1,9 GB VIRAT + 1,0 GB MEVA en
  `derivados\linea-base-frames-v1\`, pendiente de `dvc add` por Alejandro).
- Manifests/GT (sha256): manifest-virat `2d5599b0…88bff`, manifest-meva
  `33189dcd…cc41ff`, gt-virat `afacb3a4…7bf969`, gt-meva `b21e31cf…677bcb`.
- mAP con pycocotools (conf ≥ 0.001, maxDets 100, buckets COCO sobre resolución
  original), inferencia PyTorch canónica de cada candidato a 640 px. Todo
  registrado en MLflow local (experimento `linea-base-zero-shot`, 12 corridas
  `eval-*`).

## 3. Resultados VIRAT (4.632 frames, 315 vídeos, 11 escenas)

| Modelo | mAP50 | mAP50:95 | AP50 person | AP50 vehicle | AP50:95 person | AP50:95 vehicle | AP-S | AP-M | AP-L |
|---|---|---|---|---|---|---|---|---|---|
| yolo11s | 0.398 | 0.179 | 0.339 | 0.458 | 0.111 | 0.246 | 0.015 | 0.170 | 0.417 |
| **yolo11m** | **0.428** | **0.197** | 0.354 | **0.503** | 0.115 | **0.280** | 0.017 | **0.197** | 0.431 |
| rtdetrv2-s | 0.401 | 0.189 | 0.337 | 0.465 | 0.110 | 0.268 | 0.011 | 0.176 | 0.447 |
| rtdetrv2-m | 0.410 | 0.192 | 0.351 | 0.470 | 0.114 | 0.270 | 0.012 | 0.186 | **0.441** |
| dfine-s | 0.395 | 0.184 | 0.323 | 0.467 | 0.109 | 0.259 | 0.012 | 0.181 | 0.418 |
| dfine-m | 0.408 | 0.194 | **0.356** | 0.460 | **0.119** | 0.270 | 0.011 | 0.193 | 0.428 |

Desglose clase×tamaño completo en MLflow (p. ej. yolo11m: AP50 person-large 0.905,
person-small 0.087; vehicle-small ≤0.042 en todos los modelos).

## 4. Resultados MEVA (2.170 frames, 170 clips, 8 cámaras: bus G505/506/508/509, school G420/421/423/424)

| Modelo | mAP50 | mAP50:95 | AP50 person | AP50 vehicle | AP50:95 person | AP50:95 vehicle | AP-S | AP-M | AP-L |
|---|---|---|---|---|---|---|---|---|---|
| yolo11s | 0.198 | 0.088 | 0.354 | 0.041 | 0.152 | 0.025 | 0.022 | 0.088 | 0.110 |
| yolo11m | 0.208 | 0.093 | 0.375 | 0.040 | 0.161 | 0.025 | 0.016 | 0.096 | 0.112 |
| rtdetrv2-s | 0.225 | 0.102 | **0.404** | 0.046 | **0.171** | 0.032 | 0.065 | 0.096 | **0.124** |
| rtdetrv2-m | 0.219 | 0.099 | 0.396 | 0.042 | 0.170 | 0.028 | 0.050 | 0.096 | 0.120 |
| dfine-s | 0.218 | 0.100 | 0.389 | 0.047 | 0.168 | 0.032 | 0.057 | 0.096 | 0.123 |
| **dfine-m** | **0.228** | **0.106** | **0.405** | **0.051** | **0.178** | **0.034** | **0.084** | **0.108** | 0.124 |

**El AP de vehicle en MEVA (0.04–0.05 para TODOS los modelos) no mide capacidad de
detección: mide el hueco de anotación** (§7): el GT de vehicle son 514 cajas de
ámbito actividad, con los aparcamientos llenos de vehículos sin anotar.

## 5. Lectura del ranking (mAP, pendiente de latencia)

- **VIRAT**: yolo11m primero (0.428/0.197), con rtdetrv2-m y dfine-m a 1,8–2,4
  puntos de mAP50. Entre candidatos permisivos: rtdetrv2-m ≈ dfine-m.
- **MEVA**: dfine-m primero (0.228/0.106); los YOLO quedan últimos, arrastrados por
  person-small (0.048–0.077 de AP50 frente a 0.155–0.221 de los DETR).
- Patrón consistente: los DETR aguantan mejor los objetos pequeños de vigilancia;
  YOLO gana en vehículos grandes de VIRAT. Las variantes m superan a las s de su
  familia salvo rtdetrv2 en MEVA (s ligeramente mejor que m).
- En latencia (§6) los seis pasan el umbral ≥15 FPS con margen; YOLO es ~4×
  más rápido que los DETR en el mismo runtime y formato.
- La ordenación de mAP es estrecha y con el mAP absoluto deprimido por anotación
  no exhaustiva (§7): esta línea base sirve para ratificar el criterio de
  aceptación, no para proclamar un ganador.

## 6. Latencia onnxruntime-gpu (FP16, 640 px, batch 1, RTX 5070 Laptop)

Protocolo (anotación normativa 1 de REGISTRO.md): CUDAExecutionProvider verificado
activo en las 6 corridas (se aborta si cae a CPU); 300 frames reales (150 VIRAT +
150 MEVA) preprocesados en RAM; warmup 50 + 300 medidas de `session.run()`;
onnxruntime-gpu **1.27.1 build CUDA 12** (sustitución cu13→cu12 registrada en
REGISTRO.md, ejecutada por Alejandro; la build cu13 inicial no podía activar CUDA
EP sobre el stack cu128 del venv).

| Modelo | media ms | p50 ms | p95 ms | FPS (1000/media) | ≥15 FPS | prepro ref. ms* |
|---|---|---|---|---|---|---|
| yolo11s | 7.37 | 6.30 | 10.41 | **135.7** | PASA | 15.2 |
| yolo11m | 9.03 | 8.49 | 11.65 | **110.8** | PASA | 14.3 |
| rtdetrv2-s | 25.79 | 24.33 | 39.92 | **38.8** | PASA | 30.0 |
| rtdetrv2-m | 34.93 | 33.22 | 50.82 | **28.6** | PASA | 30.0 |
| dfine-s | 28.27 | 27.23 | 38.39 | **35.4** | PASA | 30.2 |
| dfine-m | 36.60 | 35.88 | 48.87 | **27.3** | PASA | 29.8 |

\* Preprocesado por frame (letterbox /255 en YOLO; processor HF en CPU en los
DETR), cronometrado UNA vez como referencia, FUERA del cronómetro de latencia.
En un pipeline real el preprocesado HF en CPU (~30 ms) sería el cuello de botella
antes que el modelo; queda anotado para la fase de integración.

Comparabilidad: el ONNX de YOLO lleva el NMS embebido en el grafo (la cifra es
end-to-end hasta cajas finales); RT-DETRv2/D-FINE son NMS-free (sus logits/cajas
finales salen del grafo). Los seis se midieron con el mismo runtime, precisión,
resolución y frames.

**Sanity check FP16 vs PyTorch FP32** (conf ≥ 0.3, 20 frames por modelo, IoU ≥ 0.5):

| Modelo | dets ref | dets ONNX | emparejadas | IoU media | Δconf media |
|---|---|---|---|---|---|
| yolo11s | 289 | 292 | 282 (97.6%) | 0.986 | 0.010 |
| yolo11m | 223 | 268 | 223 (100%) | 0.985 | 0.041 |
| rtdetrv2-s | 705 | 708 | 701 (99.4%) | 0.985 | 0.004 |
| rtdetrv2-m | 617 | 618 | 612 (99.2%) | 0.985 | 0.003 |
| dfine-s | 610 | 611 | 604 (99.0%) | 0.984 | 0.007 |
| dfine-m | 626 | 626 | 619 (98.9%) | 0.983 | 0.004 |

Los exports FP16 son fieles: ≥97.6% de las detecciones de referencia emparejadas
con IoU ~0.985. Nota puntual: el ONNX de yolo11m produce 45 detecciones extra
sobre 0.3 (ligera subida de confianzas en FP16, Δconf 0.041, la mayor de la tabla);
empareja el 100% de la referencia, así que no cambia la lectura del benchmark, y
queda anotado para la validación fina del artefacto ganador.

- Aviso de proceso (registrado en REGISTRO.md): durante el export, ultralytics
  intentó auto-instalar `onnxslim` (dependencia NO aprobada); falló (pip fuera de
  PATH) y el venv quedó intacto. El harness fija ahora `YOLO_AUTOINSTALL=false`.

## 7. Revisión visual cuantificada de falsos positivos (condición de la aprobación)

Protocolo: por dataset, los 50 FPs de mayor confianza del candidato de mAP más
alto (VIRAT: yolo11m; MEVA: dfine-m), FP = detección sin GT de su clase a IoU≥0.5,
clasificación manual sobre hojas de contacto con el GT y las cajas de tipos
descartados superpuestos (`derivados\…\revision-fps\`, clasificacion.json).

| Dataset | Modelo | Positivo sin etiquetar | Error real | Ambiguo | % sin etiquetar |
|---|---|---|---|---|---|
| VIRAT | yolo11m | **50** | 0 | 0 | **100%** |
| MEVA | dfine-m | **50** | 0 | 0 | **100%** |

**DESCARGO EXPLÍCITO (umbral del 30% superado con creces): el mAP absoluto de esta
línea base está sistemáticamente deprimido por positivos sin etiquetar.** Los 50 FPs
de VIRAT son vehículos reales aparcados sin caja GT (≈3–4 objetos físicos repetidos
en 7 vídeos de las escenas 0502/0503/0401, conf 0.92–0.93); los 50 de MEVA, vehículos
aparcados sin GT en el parking de las cámaras school (conf 0.96–0.97). Ninguno solapa
cajas de tipos descartados (tipo 0/4/5, other/bag) a IoU≥0.3: son huecos de anotación
puros, no efecto de nuestros descartes. Consecuencia direccional: el sesgo castiga
MÁS al detector con mejor recall real; las comparaciones relativas siguen siendo
informativas, los valores absolutos NO son cotas de rendimiento real.

## 8. Tipo 0 de VIRAT (hallazgo de los smoke tests, cuantificado)

416.698 filas tipo 0 (no documentado en el README) concentradas en **7 vídeos**:
VIRAT_S_000003 (188k), 000001 (124k), 000002 (71k), 000206_07 (22k) y tres de la
escena 0502/0503 (<8k). Además ~12 filas con tipos basura (142, 9076…), descartadas
y contadas. Detalle por vídeo en `derivados\…\analisis-tipo0-virat.json`. En la
revisión de FPs ningún FP top-50 solapó cajas tipo 0; el riesgo de positivos sin
etiquetar por tipo 0 queda acotado a esos 7 vídeos (refuerzo de la advertencia 2).

## 9. Advertencias asumidas, con números finales

1. **Anotación no exhaustiva**: confirmada y cuantificada (§7, 100% en ambos top-50).
2. **Huecos VIRAT**: 14 vídeos sin anotación excluidos; tipo 0 en 7 vídeos (§8).
3. **Bicis/motos fuera por ambos lados**: descartes simétricos aplicados (6.763
   bicycle + 3.207 motorcycle descartadas solo en las predicciones de yolo11s-VIRAT,
   como referencia de volumen; conteos por corrida en MLflow).
4. **`other`/`bag` de MEVA descartados**: sin solape con los FPs top (§7).
5. **Objetos pequeños a 640 px**: AP-small 0.011–0.022 (VIRAT) y 0.016–0.084 (MEVA)
   para todos; es el punto de operación de §15, no un defecto del harness.
6. **GPU de portátil**: medido enchufado y en perfil de máximo rendimiento; el p95
   se reporta junto a la media (§6; p95/p50 ≈ 1.4–1.6 en los DETR, huella térmica
   de portátil). La latencia en el hardware final del producto se re-medirá.
7. **FP16**: sanity check ejecutado (§6): ≥97.6% emparejadas, IoU ~0.985; el ONNX
   de yolo11m sube ligeramente las confianzas (Δ 0.041, 45 dets extra a conf 0.3).
8. **mAP en PyTorch / latencia en ONNX**: mantenido; el sanity check del punto 7
   evidencia que los artefactos exportados son fieles al modelo evaluado.

## 10. Elementos para la ratificación del criterio de aceptación (§15)

Sin decidir nada aquí: (a) el mAP50 zero-shot en vigilancia real queda en 0.40–0.43
(VIRAT) y 0.20–0.23 (MEVA) con anotación no exhaustiva; (b) el orden entre
candidatos es estrecho y cambia según dataset (yolo11m lidera VIRAT; dfine-m lidera
MEVA con los DETR claramente mejores en person-small); (c) los valores absolutos NO
son cotas de rendimiento real (§7); (d) los seis candidatos PASAN el umbral de
latencia de §15 (≥15 FPS en onnxruntime-gpu FP16 640 px: YOLO 111–136 FPS, DETR
27–39 FPS en la RTX 5070 Laptop), así que la latencia no elimina a ningún candidato
en esta GPU y el margen de cada uno queda en la tabla de §6. La ratificación del
criterio y cualquier lectura de ganador corresponden a Alejandro.

## Reproducibilidad

MLflow local (`mlflow.db`, sqlite): experimento `linea-base-zero-shot`, 18 corridas
— 12 `eval-{dataset}-{modelo}` + 6 `lat-{modelo}` — con params (semilla 20260716,
config, sha256 de pesos/manifests/GT/predicciones/ONNX, snapshots HF, md5 DVC,
provider y dtype de entrada en las corridas de latencia, versiones: torch
2.11.0+cu128, transformers 4.57.6, ultralytics 8.3.253, onnxruntime-gpu 1.27.1
build cu12, opencv 4.13.0, numpy 2.4.4, pycocotools 2.0.11), métricas completas
por clase×tamaño, latencias y sanity FP16, y artefactos (resumen COCOeval,
descartes por clase COCO). Predicciones y derivados en
`PROVIDENCE-DATA\derivados\linea-base-frames-v1\` (congela Alejandro con `dvc add`).
