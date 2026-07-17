"""Referencia de preprocesado+inferencia para C1 (producto: sidecar detector).

Genera el GOLDEN contra el que el pipeline propio del sidecar (PyAV/sws +
numpy + onnxruntime) se verifica: los mismos golden frames pasados por el
pipeline de REFERENCIA de B3 (processor HF canónico → ONNX dfine-m FP16 →
decodificación espejo de post_process_object_detection).

El artefacto va a PROVIDENCE-DATA (regla 6); el repo del producto solo lleva
el test que lo consume y el informe en §23.

uso (venv del sandbox, HF_HOME apuntando a la caché local):
  python scripts/genera_referencia_c1.py
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault(
    "HF_HOME", r"C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA\pesos\linea-base\hf-cache"
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.latencia import (  # noqa: E402
    cargar_sesion,
    decodificar_hf_onnx,
    dtype_de_entrada,
)

HF_ID = "ustc-community/dfine-medium-coco"
ONNX = Path(r"C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA\derivados\linea-base-frames-v1\onnx\dfine-m.onnx")
FRAMES = Path(r"C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA\derivados\c1-referencia-preprocesado\frames")
SALIDA = Path(r"C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA\derivados\c1-referencia-preprocesado")
CONF_MIN = 0.30  # margen bajo el umbral de operación (0.5) para diagnóstico


def main() -> int:
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(HF_ID, local_files_only=True)
    sesion = cargar_sesion(ONNX)
    dtype = dtype_de_entrada(sesion)
    nombre_entrada = sesion.get_inputs()[0].name
    print(f"referencia C1: {ONNX.name}, entrada '{nombre_entrada}' {dtype}")

    resultado = {}
    resumen = []
    for ruta in sorted(FRAMES.glob("*.png")):
        bgr = cv2.imread(str(ruta))
        if bgr is None:
            raise RuntimeError(f"no puedo leer {ruta}")
        alto, ancho = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pv = processor(images=[rgb], return_tensors="np")["pixel_values"].astype(dtype)
        t0 = time.perf_counter()
        salida = sesion.run(None, {nombre_entrada: np.ascontiguousarray(pv)})
        ms = (time.perf_counter() - t0) * 1000.0
        dets = decodificar_hf_onnx(salida, alto, ancho, conf_min=CONF_MIN)
        cls = np.array([d[0] for d in dets], dtype=np.int32)
        conf = np.array([d[1] for d in dets], dtype=np.float32)
        xyxy = np.array([[d[2], d[3], d[4], d[5]] for d in dets], dtype=np.float32).reshape(-1, 4)
        clave = ruta.stem
        resultado[f"{clave}_cls"] = cls
        resultado[f"{clave}_conf"] = conf
        resultado[f"{clave}_xyxy"] = xyxy
        personas_05 = int(((cls == 0) & (conf >= 0.5)).sum())
        total_05 = int((conf >= 0.5).sum())
        resumen.append((clave, total_05, personas_05, ms))
        print(f"  {clave}: {total_05} dets >=0.5 ({personas_05} person), inferencia {ms:.1f} ms")

    SALIDA.mkdir(parents=True, exist_ok=True)
    np.savez(SALIDA / "referencia-dfine-m.npz", **resultado)
    meta = {
        "hf_id": HF_ID,
        "onnx": str(ONNX),
        "conf_min": CONF_MIN,
        "frames": [r[0] for r in resumen],
        "generado": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nota": "pipeline de referencia B3: processor HF canónico + ORT CUDA + "
        "decodificar_hf_onnx (espejo de post_process_object_detection)",
    }
    (SALIDA / "referencia-dfine-m.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"OK: {SALIDA / 'referencia-dfine-m.npz'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
