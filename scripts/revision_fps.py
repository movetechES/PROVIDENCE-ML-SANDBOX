"""Revisión visual cuantificada de falsos positivos (condición de la aprobación B3).

Por dataset: los 50 FPs de mayor confianza del candidato de mAP más alto.
FP = detección L1 sin pareja GT de su clase a IoU>=0.5 (emparejado greedy por
confianza, uno a uno). Cada FP se cruza además con las cajas ORIGINALES de tipos
descartados (VIRAT: tipo 0/4/5; MEVA: other/bag): el solape a IoU>=0.3 se marca
como evidencia de "positivo sin etiquetar" para la clasificación manual.

Salida: crops individuales + hojas de contacto numeradas (2x5) en
derivados/revision-fps/{dataset}/ y un JSON con la metainformación por FP.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ_SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_SANDBOX))

from harness import rutas  # noqa: E402
from harness.conversores.meva_a_coco import convertir_clip  # noqa: E402
from harness.ontologia import MEVA_A_L1, VIRAT_A_L1  # noqa: E402

DERIVADOS = rutas.DERIVADOS
COLOR_FP = (0, 0, 255)        # rojo: la detección FP
COLOR_GT_MISMA = (0, 200, 0)  # verde: GT de la misma clase
COLOR_GT_OTRA = (255, 200, 0) # cian-ish: GT de la otra clase
COLOR_DESCARTADO = (0, 220, 220)  # amarillo: caja de tipo descartado


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if not inter:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _xywh_a_xyxy(b):
    return (b[0], b[1], b[0] + b[2], b[1] + b[3])


def cajas_descartadas_virat(video_id, frame, cache={}):
    if video_id not in cache:
        cajas = {}
        ruta = rutas.VIRAT_ANOTACIONES / f"{video_id}.viratdata.objects.txt"
        with open(ruta, encoding="utf-8") as f:
            for linea in f:
                p = linea.split()
                if len(p) != 8:
                    continue
                tipo = int(p[7])
                if tipo in VIRAT_A_L1:
                    continue
                fr = int(p[2])
                x, y, w, h = (int(v) for v in p[3:7])
                cajas.setdefault(fr, []).append((f"tipo{tipo}", (x, y, x + w, y + h)))
        cache[video_id] = cajas
    return cache[video_id].get(frame, [])


def cajas_descartadas_meva(clip, frame, ancho, alto, cache={}):
    if clip not in cache:
        import yaml

        from harness.conversores.meva_a_coco import _cargar_kpf, _indice_tipos, localizar_anotacion

        _, ruta_geom = localizar_anotacion(clip)
        tipos = _indice_tipos(ruta_geom.with_name(ruta_geom.name.replace(".geom.yml", ".types.yml")))
        cajas = {}
        for entrada in _cargar_kpf(ruta_geom):
            geom = entrada.get("geom") if isinstance(entrada, dict) else None
            if not geom:
                continue
            etiqueta = tipos.get(geom["id1"])
            if etiqueta is None or etiqueta in MEVA_A_L1:
                continue
            x1, y1, x2, y2 = (float(v) for v in str(geom["g0"]).split())
            cajas.setdefault(int(geom["ts0"]), []).append((etiqueta, (x1, y1, x2, y2)))
        cache[clip] = cajas
    return cache[clip].get(frame, [])


def encontrar_fps(dataset, modelo_id, n=50, iou_min=0.5):
    with open(DERIVADOS / f"gt-{dataset}.json", encoding="utf-8") as f:
        gt = json.load(f)
    with open(DERIVADOS / f"manifest-{dataset}.json", encoding="utf-8") as f:
        manifest = {fr["image_id"]: fr for fr in json.load(f)["frames"]}
    with open(DERIVADOS / "predicciones" / f"{dataset}-{modelo_id}.json", encoding="utf-8") as f:
        preds = json.load(f)

    gt_por_imagen = {}
    for a in gt["annotations"]:
        gt_por_imagen.setdefault(a["image_id"], []).append(a)

    fps = []
    preds_por_imagen = {}
    for p in preds:
        preds_por_imagen.setdefault(p["image_id"], []).append(p)
    for image_id, plist in preds_por_imagen.items():
        gts = gt_por_imagen.get(image_id, [])
        usadas = set()
        for p in sorted(plist, key=lambda q: -q["score"]):
            caja_p = _xywh_a_xyxy(p["bbox"])
            mejor, mejor_iou = None, iou_min
            iou_max = 0.0
            for i, g in enumerate(gts):
                if g["category_id"] != p["category_id"]:
                    continue
                v = _iou(caja_p, _xywh_a_xyxy(g["bbox"]))
                iou_max = max(iou_max, v)
                if i not in usadas and v >= mejor_iou:
                    mejor, mejor_iou = i, v
            if mejor is not None:
                usadas.add(mejor)
            else:
                fps.append({**p, "iou_max_gt_misma_clase": round(iou_max, 3)})
    fps.sort(key=lambda p: -p["score"])
    return fps[:n], gt_por_imagen, manifest


def render(dataset, modelo_id, n=50):
    fps, gt_por_imagen, manifest = encontrar_fps(dataset, modelo_id, n)
    dir_salida = DERIVADOS / "revision-fps" / dataset
    dir_salida.mkdir(parents=True, exist_ok=True)
    nombres_cat = {1: "person", 2: "vehicle"}
    tiles, meta = [], []

    for rango, fp in enumerate(fps, 1):
        fr = manifest[fp["image_id"]]
        imagen = cv2.imread(str(DERIVADOS / "frames" / fr["file_name"]))
        x1, y1, x2, y2 = (int(v) for v in _xywh_a_xyxy(fp["bbox"]))

        if dataset == "virat":
            descartadas = cajas_descartadas_virat(fr["unidad"], fr["frame"])
        else:
            descartadas = cajas_descartadas_meva(fr["unidad"], fr["frame"], fr["width"], fr["height"])
        solape = ""
        for etiqueta, caja in descartadas:
            if _iou((x1, y1, x2, y2), caja) >= 0.3:
                solape = etiqueta
                break

        for g in gt_por_imagen.get(fp["image_id"], []):
            gx1, gy1, gx2, gy2 = (int(v) for v in _xywh_a_xyxy(g["bbox"]))
            color = COLOR_GT_MISMA if g["category_id"] == fp["category_id"] else COLOR_GT_OTRA
            cv2.rectangle(imagen, (gx1, gy1), (gx2, gy2), color, 2)
        for _, caja in descartadas:
            cx1, cy1, cx2, cy2 = (int(v) for v in caja)
            cv2.rectangle(imagen, (cx1, cy1), (cx2, cy2), COLOR_DESCARTADO, 2)
        cv2.rectangle(imagen, (x1, y1), (x2, y2), COLOR_FP, 3)

        cw, ch = x2 - x1, y2 - y1
        margen = max(int(max(cw, ch) * 1.25), 120)
        rx1, ry1 = max(0, x1 - margen), max(0, y1 - margen)
        rx2, ry2 = min(imagen.shape[1], x2 + margen), min(imagen.shape[0], y2 + margen)
        crop = imagen[ry1:ry2, rx1:rx2]

        lado = 360
        escala = min(lado / crop.shape[0], lado / crop.shape[1])
        crop = cv2.resize(crop, (int(crop.shape[1] * escala), int(crop.shape[0] * escala)))
        tile = np.full((lado + 42, lado, 3), 30, np.uint8)
        oy, ox = (lado - crop.shape[0]) // 2, (lado - crop.shape[1]) // 2
        tile[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
        texto = f"#{rango} {nombres_cat[fp['category_id']]} {fp['score']:.2f}"
        if solape:
            texto += f" [{solape}]"
        cv2.putText(tile, texto, (6, lado + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(
            tile,
            f"{fr['unidad'][:38]} f{fr['frame']}",
            (6, lado + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )
        tiles.append(tile)
        cv2.imwrite(str(dir_salida / f"fp-{rango:02d}.jpg"), tile)
        meta.append(
            {
                "rango": rango,
                "clase": nombres_cat[fp["category_id"]],
                "score": fp["score"],
                "unidad": fr["unidad"],
                "frame": fr["frame"],
                "bbox": fp["bbox"],
                "iou_max_gt_misma_clase": fp["iou_max_gt_misma_clase"],
                "solapa_tipo_descartado": solape or None,
            }
        )

    for hoja in range((len(tiles) + 9) // 10):
        grupo = tiles[hoja * 10 : hoja * 10 + 10]
        while len(grupo) < 10:
            grupo.append(np.full_like(tiles[0], 30))
        filas = [np.hstack(grupo[i * 5 : i * 5 + 5]) for i in range(2)]
        cv2.imwrite(str(dir_salida / f"hoja-{hoja + 1}.jpg"), np.vstack(filas))

    with open(dir_salida / f"fps-{dataset}-{modelo_id}.json", "w", encoding="utf-8") as f:
        json.dump({"modelo": modelo_id, "dataset": dataset, "fps": meta}, f, indent=1)
    auto = sum(1 for m in meta if m["solapa_tipo_descartado"])
    print(
        f"[{dataset}/{modelo_id}] {len(meta)} FPs renderizados en {dir_salida} | "
        f"con solape a tipo descartado (IoU>=0.3): {auto}"
    )


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
