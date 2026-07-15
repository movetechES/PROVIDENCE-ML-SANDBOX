"""Exportación a ONNX y medición de latencia en onnxruntime-gpu (FP16, 640 px).

Anotación normativa 1 de REGISTRO.md: la latencia del benchmark se mide SIEMPRE
en onnxruntime-gpu porque ONNX es el formato que cruza al producto. Se exige
CUDAExecutionProvider (se aborta si cae a CPU). Se cronometra session.run()
(warmup + N medidas) sobre frames reales preprocesados en RAM; el preprocesado
se cronometra aparte, una vez, como referencia.

torch se importa antes que onnxruntime: sus DLLs de CUDA/cuDNN resuelven las
que onnxruntime-gpu necesita en Windows.
"""

import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch


# ---------- Exportación ----------

def exportar_yolo_onnx(ruta_pesos, dir_salida, imgsz=640):
    """Export ultralytics con NMS embebido en el grafo y FP16."""
    from ultralytics import YOLO

    modelo = YOLO(str(ruta_pesos))
    ruta = modelo.export(
        format="onnx", imgsz=imgsz, half=True, nms=True, device=0, batch=1, dynamic=False
    )
    dir_salida.mkdir(parents=True, exist_ok=True)
    destino = dir_salida / Path(ruta).name
    if Path(ruta).resolve() != destino.resolve():
        shutil.move(str(ruta), str(destino))
    return destino


class _EnvolturaHF(torch.nn.Module):
    """Devuelve (logits, pred_boxes) como tupla de tensores exportable."""

    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo

    def forward(self, pixel_values):
        salida = self.modelo(pixel_values=pixel_values)
        return salida.logits, salida.pred_boxes


def exportar_hf_onnx(hf_id, dir_salida, nombre, imgsz=640, bloquear_gridsample=False):
    """torch.onnx.export FP32 + conversión FP16 (onnxruntime.transformers.float16).

    Si el grafo FP16 falla en tiempo de sesión/run, reintentar con
    bloquear_gridsample=True (GridSample queda en FP32 con casts).
    """
    import onnx
    from onnxruntime.transformers import float16 as f16
    from transformers import AutoModelForObjectDetection

    modelo = AutoModelForObjectDetection.from_pretrained(hf_id, local_files_only=True).eval()
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta32 = dir_salida / f"{nombre}-fp32.onnx"
    torch.onnx.export(
        _EnvolturaHF(modelo),
        (dummy,),
        str(ruta32),
        input_names=["pixel_values"],
        output_names=["logits", "pred_boxes"],
        opset_version=17,
        dynamo=False,
    )

    kwargs = {"keep_io_types": False}
    if bloquear_gridsample:
        base = list(getattr(f16, "DEFAULT_OP_BLOCK_LIST", []))
        kwargs["op_block_list"] = base + ["GridSample"]
    modelo16 = f16.convert_float_to_float16(onnx.load(str(ruta32)), **kwargs)
    ruta16 = dir_salida / f"{nombre}.onnx"
    onnx.save(modelo16, str(ruta16))
    return ruta16, ruta32


# ---------- Sesión y medición ----------

def cargar_sesion(ruta_onnx):
    import onnxruntime as ort

    sesion = ort.InferenceSession(str(ruta_onnx), providers=["CUDAExecutionProvider"])
    activos = sesion.get_providers()
    if activos[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDAExecutionProvider no activo (providers: {activos})")
    return sesion


def medir(sesion, entradas, warmup, medidas):
    nombre = sesion.get_inputs()[0].name
    for i in range(warmup):
        sesion.run(None, {nombre: entradas[i % len(entradas)]})
    tiempos = []
    for i in range(medidas):
        x = entradas[i % len(entradas)]
        t0 = time.perf_counter()
        sesion.run(None, {nombre: x})
        tiempos.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(tiempos)
    return {
        "ms_media": float(arr.mean()),
        "ms_p50": float(np.percentile(arr, 50)),
        "ms_p95": float(np.percentile(arr, 95)),
        "fps_media": float(1000.0 / arr.mean()),
        "n_medidas": int(arr.size),
    }


# ---------- Preprocesado (fuera del cronómetro) ----------

def letterbox(bgr, lado=640, relleno=114):
    alto, ancho = bgr.shape[:2]
    escala = min(lado / alto, lado / ancho)
    nuevo_w, nuevo_h = round(ancho * escala), round(alto * escala)
    redim = cv2.resize(bgr, (nuevo_w, nuevo_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (lado - nuevo_w) / 2, (lado - nuevo_h) / 2
    arriba, abajo = round(dh - 0.1), round(dh + 0.1)
    izq, der = round(dw - 0.1), round(dw + 0.1)
    lienzo = cv2.copyMakeBorder(
        redim, arriba, abajo, izq, der, cv2.BORDER_CONSTANT, value=(relleno,) * 3
    )
    return lienzo, escala, (izq, arriba)


def preprocesar_yolo(bgr, lado, dtype):
    lienzo, escala, despl = letterbox(bgr, lado)
    x = lienzo[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x.astype(dtype)), escala, despl


def preprocesar_hf(processor, bgr, dtype):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pv = processor(images=[rgb], return_tensors="np")["pixel_values"]
    return np.ascontiguousarray(pv.astype(dtype))


def dtype_de_entrada(sesion):
    tipo = sesion.get_inputs()[0].type
    return np.float16 if "float16" in tipo else np.float32


# ---------- Decodificación para el sanity check FP16 vs FP32 ----------

def decodificar_yolo_onnx(salida, escala, despl, conf_min=0.3):
    """Salida con NMS embebido: (1, N, 6) = x1,y1,x2,y2,conf,cls en coords letterbox."""
    dets = []
    izq, arriba = despl
    for x1, y1, x2, y2, conf, cls in np.asarray(salida[0], dtype=np.float32)[0]:
        if conf < conf_min:
            continue
        dets.append(
            (
                int(cls),
                float(conf),
                (x1 - izq) / escala,
                (y1 - arriba) / escala,
                (x2 - izq) / escala,
                (y2 - arriba) / escala,
            )
        )
    return dets


def decodificar_hf_onnx(salida, alto, ancho, conf_min=0.3, k=300):
    """Espeja post_process_object_detection: sigmoid + top-k aplanado + cxcywh->xyxy."""
    logits = np.asarray(salida[0], dtype=np.float32)[0]
    cajas = np.asarray(salida[1], dtype=np.float32)[0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    plano = probs.reshape(-1)
    k = min(k, plano.size)
    idx = np.argpartition(-plano, k - 1)[:k]
    dets = []
    n_clases = probs.shape[1]
    for i in idx:
        conf = float(plano[i])
        if conf < conf_min:
            continue
        q, cls = divmod(int(i), n_clases)
        cx, cy, w, h = cajas[q]
        dets.append(
            (
                cls,
                conf,
                (cx - w / 2) * ancho,
                (cy - h / 2) * alto,
                (cx + w / 2) * ancho,
                (cy + h / 2) * alto,
            )
        )
    return dets


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter) if inter else 0.0


def comparar_detecciones(dets_ref, dets_onnx, iou_min=0.5):
    """Empareja por clase+IoU (greedy por confianza). Referencia: PyTorch FP32."""
    libres = list(range(len(dets_onnx)))
    ious, dconfs = [], []
    for cls, conf, *caja in sorted(dets_ref, key=lambda d: -d[1]):
        mejor, mejor_iou = None, iou_min
        for j in libres:
            ocls, oconf, *ocaja = dets_onnx[j]
            if ocls != cls:
                continue
            v = _iou(caja, ocaja)
            if v >= mejor_iou:
                mejor, mejor_iou = j, v
        if mejor is not None:
            libres.remove(mejor)
            ious.append(mejor_iou)
            dconfs.append(abs(conf - dets_onnx[mejor][1]))
    return {
        "n_ref": len(dets_ref),
        "n_onnx": len(dets_onnx),
        "emparejadas": len(ious),
        "iou_media": float(np.mean(ious)) if ious else float("nan"),
        "delta_conf_media": float(np.mean(dconfs)) if dconfs else float("nan"),
    }
