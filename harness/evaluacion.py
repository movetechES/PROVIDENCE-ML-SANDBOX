"""Evaluación mAP COCO (pycocotools): global, por clase y por bucket de tamaño.

Buckets estándar COCO sobre la resolución ORIGINAL: small <32^2, medium
32^2..96^2, large >96^2 px. maxDets estándar [1, 10, 100].
"""

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

AREAS = ("all", "small", "medium", "large")


def _ap_de_precision(precision, t_idx=None):
    """AP medio de un sub-tensor de precision de COCOeval, ignorando -1."""
    if t_idx is not None:
        precision = precision[t_idx]
    validas = precision[precision > -1]
    return float(validas.mean()) if validas.size else float("nan")


def evaluar(ruta_gt, predicciones):
    """predicciones: lista COCO de dicts {image_id, category_id, bbox, score}.

    Devuelve (metricas: dict plano para MLflow, resumen_texto de COCOeval).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(ruta_gt))
        if not predicciones:
            raise ValueError("Sin predicciones que evaluar")
        dt = gt.loadRes(predicciones)
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()

    salida = io.StringIO()
    with contextlib.redirect_stdout(salida):
        ev.summarize()

    m = {
        "mAP50_95": float(ev.stats[0]),
        "mAP50": float(ev.stats[1]),
        "AP_small": float(ev.stats[3]),
        "AP_medium": float(ev.stats[4]),
        "AP_large": float(ev.stats[5]),
        "AR100": float(ev.stats[8]),
    }

    # Desglose por clase y clase x tamaño desde el tensor de precisión [T,R,K,A,M].
    precision = ev.eval["precision"]
    m_idx = -1  # maxDets = 100
    nombres = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    for k, cat_id in enumerate(ev.params.catIds):
        nombre = nombres[cat_id]
        for a, area in enumerate(AREAS):
            sub = precision[:, :, k, a, m_idx]
            sufijo = "" if area == "all" else f"_{area}"
            m[f"AP50_95_{nombre}{sufijo}"] = _ap_de_precision(sub)
            m[f"AP50_{nombre}{sufijo}"] = _ap_de_precision(sub, t_idx=0)
    return m, salida.getvalue()


def a_predicciones_coco(detecciones_por_imagen, mapeo_l1):
    """(image_id -> [(nombre_clase, conf, x, y, w, h)]) -> lista COCO de dets.

    Mapea por NOMBRE de clase con `mapeo_l1` (ontologia.COCO_A_L1); las clases
    fuera del mapeo se descartan (simétrico con el GT). Devuelve también el
    contador de descartes por clase.
    """
    from collections import Counter

    dets, descartes = [], Counter()
    for image_id, detecciones in detecciones_por_imagen.items():
        for nombre, conf, x, y, w, h in detecciones:
            categoria = mapeo_l1.get(nombre)
            if categoria is None:
                descartes[nombre] += 1
                continue
            dets.append(
                {
                    "image_id": image_id,
                    "category_id": categoria,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "score": round(conf, 5),
                }
            )
    return dets, descartes
