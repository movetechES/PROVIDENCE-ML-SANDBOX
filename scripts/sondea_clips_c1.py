"""Sondeo de idoneidad de clips para C1: ¿qué ve dfine-m en cada candidato?

Lee frames directamente de los .avi originales (cv2), pasa el pipeline de
referencia y cuenta detecciones por clase — el clip del gate necesita
PERSONAS visibles de forma sostenida.
"""

import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", r"C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA\pesos\linea-base\hf-cache"
)

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.latencia import cargar_sesion, decodificar_hf_onnx, dtype_de_entrada

from transformers import AutoImageProcessor

BASE = Path(r"C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA\externos\meva\video\2018-03-11\11")
CLIPS = [
    "2018-03-11.11-20-01.11-25-01.school.G424.r13.avi",
    "2018-03-11.11-20-01.11-25-01.bus.G505.r13.avi",
    "2018-03-11.11-20-00.11-25-00.bus.G506.r13.avi",
    "2018-03-11.11-20-00.11-25-00.school.G421.r13.avi",
    "2018-03-11.11-20-00.11-25-00.school.G420.r13.avi",
]
SEGUNDOS = [20, 80, 150, 220, 280]
COCO = {0: "person", 1: "bicycle", 2: "car", 5: "bus", 7: "truck"}

processor = AutoImageProcessor.from_pretrained(
    "ustc-community/dfine-medium-coco", local_files_only=True
)
sesion = cargar_sesion(
    Path(r"C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA\derivados\linea-base-frames-v1\onnx\dfine-m.onnx")
)
dtype = dtype_de_entrada(sesion)

for clip in CLIPS:
    ruta = BASE / clip
    cap = cv2.VideoCapture(str(ruta))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = Counter()
    personas_por_frame = []
    for s in SEGUNDOS:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pv = processor(images=[rgb], return_tensors="np")["pixel_values"].astype(dtype)
        salida = sesion.run(None, {"pixel_values": np.ascontiguousarray(pv)})
        dets = decodificar_hf_onnx(salida, bgr.shape[0], bgr.shape[1], conf_min=0.5)
        for cls, conf, *_ in dets:
            total[COCO.get(cls, f"c{cls}")] += 1
        personas_por_frame.append(sum(1 for d in dets if d[0] == 0))
    cap.release()
    print(f"{clip}: person/frame={personas_por_frame} | total={dict(total)}")
