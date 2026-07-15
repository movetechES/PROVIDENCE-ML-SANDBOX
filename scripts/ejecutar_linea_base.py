"""Orquestador de la línea base zero-shot (B3, fase 2).

Subcomandos: preparar | evaluar | exportar | latencia | tipo0-virat
Configuración del protocolo: configs/linea-base.yaml (aprobada 2026-07-16).
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

RAIZ_SANDBOX = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ_SANDBOX))

from harness import rutas  # noqa: E402

# La caché HF vive en PROVIDENCE-DATA; local_files_only en los wrappers impide
# descargas. Debe fijarse ANTES de importar transformers.
os.environ.setdefault("HF_HOME", str(rutas.PESOS / "hf-cache"))
# Ultralytics no instala nada por su cuenta (aviso registrado en REGISTRO.md
# 2026-07-16: intento de autoinstall de onnxslim durante el export de B3).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from harness.conversores.comun import ConstructorCocoGT  # noqa: E402
from harness.conversores.meva_a_coco import convertir_clip, localizar_anotacion  # noqa: E402
from harness.conversores.virat_a_coco import convertir_video  # noqa: E402
from harness.extraccion import contar_frames, extraer_frames  # noqa: E402
from harness.muestreo import seleccionar_frames  # noqa: E402
from harness.ontologia import COCO_A_L1, VIRAT_A_L1  # noqa: E402

DERIVADOS = rutas.DERIVADOS
DIR_FRAMES = DERIVADOS / "frames"
DIR_PREDICCIONES = DERIVADOS / "predicciones"
DIR_ONNX = DERIVADOS / "onnx"


def cargar_config():
    import yaml

    with open(RAIZ_SANDBOX / "configs" / "linea-base.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_modelo(cfg, modelo_id):
    for m in cfg["modelos"]:
        if m["id"] == modelo_id:
            return m
    raise SystemExit(f"Modelo desconocido: {modelo_id}")


def listar_videos_virat():
    videos = []
    for ruta in sorted(rutas.VIRAT_VIDEOS.glob("*.mp4")):
        anot = rutas.VIRAT_ANOTACIONES / f"{ruta.stem}.viratdata.objects.txt"
        if anot.is_file():
            videos.append(ruta)
    return videos


def listar_clips_meva():
    with open(rutas.MEVA_SUBCONJUNTO, encoding="utf-8-sig") as f:
        return [linea.strip() for linea in f if linea.strip()]


def localizar_video_meva(clip):
    fecha = clip.split(".")[0]
    candidatos = sorted((rutas.MEVA_VIDEOS / fecha).rglob(f"{clip}*.avi"))
    if not candidatos:
        raise FileNotFoundError(f"Sin video para {clip}")
    return candidatos[0]


# ---------- preparar ----------

def preparar_dataset(dataset, cfg):
    p = cfg["muestreo"][dataset]
    zancada = p["zancada_min_frames"]
    tope = p.get("tope_por_video") or p.get("tope_por_clip")
    jpeg_calidad = cfg["jpeg_calidad"]

    constructor = ConstructorCocoGT()
    manifest = []
    resumen = Counter()
    t0 = time.time()

    if dataset == "virat":
        unidades = [(v.stem, v) for v in listar_videos_virat()]
    else:
        unidades = [(c, localizar_video_meva(c)) for c in listar_clips_meva()]

    for nombre, ruta_video in unidades:
        n_frames, ancho, alto = contar_frames(ruta_video)
        if dataset == "virat":
            res = convertir_video(nombre, ancho, alto)
        else:
            res = convertir_clip(nombre, ancho, alto)
            if res.marcado_vacio:
                resumen["unidades_vacias"] += 1
        elegibles = list(res.cajas_por_frame)
        seleccion = seleccionar_frames(elegibles, n_frames, zancada, tope)
        resumen["unidades"] += 1
        if not seleccion:
            resumen["unidades_sin_seleccion"] += 1
            continue
        extraidos, fallidos = extraer_frames(
            ruta_video, seleccion, DIR_FRAMES / dataset, nombre, jpeg_calidad
        )
        resumen["frames_fallidos"] += len(fallidos)
        for frame, fichero in extraidos:
            cajas = res.cajas_por_frame[frame]
            image_id = constructor.agregar_frame(f"{dataset}/{fichero}", ancho, alto, cajas)
            manifest.append(
                {
                    "image_id": image_id,
                    "file_name": f"{dataset}/{fichero}",
                    "unidad": nombre,
                    "frame": frame,
                    "width": ancho,
                    "height": alto,
                    "n_gt": len(cajas),
                }
            )
            resumen["frames"] += 1
            resumen["cajas_gt"] += len(cajas)

    por_clase = Counter(a["category_id"] for a in constructor.coco["annotations"])
    doc_manifest = {
        "version": cfg["version"],
        "dataset": dataset,
        "muestreo": {"zancada_min_frames": zancada, "tope": tope, "semilla": cfg["semilla"]},
        "jpeg_calidad": jpeg_calidad,
        "resumen": dict(resumen) | {"gt_person": por_clase.get(1, 0), "gt_vehicle": por_clase.get(2, 0)},
        "frames": manifest,
    }
    DERIVADOS.mkdir(parents=True, exist_ok=True)
    with open(DERIVADOS / f"manifest-{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(doc_manifest, f, indent=1)
    with open(DERIVADOS / f"gt-{dataset}.json", "w", encoding="utf-8") as f:
        json.dump(constructor.coco, f)
    print(f"[{dataset}] {dict(resumen)} | GT por clase: {dict(por_clase)} | {time.time()-t0:.0f}s")


# ---------- evaluar ----------

def crear_detector(mc, cfg):
    if mc["candidato"] == "yolo-ultralytics":
        from harness.inferencia.yolo_ultralytics import DetectorYolo

        return DetectorYolo(
            rutas.PROVIDENCE_DATA / mc["pesos"], imgsz=cfg["imgsz"], conf=cfg["conf_min"]
        )
    from harness.inferencia.hf_detr import DetectorHF

    return DetectorHF(mc["hf_id"], conf=cfg["conf_min"])


def info_pesos(mc):
    from harness.registro_mlflow import sha256_fichero

    if mc["candidato"] == "yolo-ultralytics":
        ruta = rutas.PROVIDENCE_DATA / mc["pesos"]
        return {"pesos": mc["pesos"], "pesos_sha256": sha256_fichero(ruta)}
    repo = "models--" + mc["hf_id"].replace("/", "--")
    snapshots = sorted((rutas.PESOS / "hf-cache" / "hub" / repo / "snapshots").iterdir())
    snapshot = snapshots[0].name
    return {
        "pesos": mc["hf_id"],
        "hf_snapshot": snapshot,
        "pesos_sha256": sha256_fichero(snapshots[0] / "model.safetensors"),
    }


def evaluar_modelo(modelo_id, dataset, cfg):
    from harness.evaluacion import a_predicciones_coco, evaluar
    from harness.registro_mlflow import params_comunes, registrar_corrida, sha256_fichero

    mc = config_modelo(cfg, modelo_id)
    with open(DERIVADOS / f"manifest-{dataset}.json", encoding="utf-8") as f:
        manifest = json.load(f)
    ruta_gt = DERIVADOS / f"gt-{dataset}.json"

    detector = crear_detector(mc, cfg)
    lote_max = 16 if mc["candidato"] == "yolo-ultralytics" else 8
    frames = manifest["frames"]
    detecciones = {}
    t0 = time.time()
    for i in range(0, len(frames), lote_max):
        lote = frames[i : i + lote_max]
        rutasl = [DIR_FRAMES / fr["file_name"] for fr in lote]
        for fr, dets in zip(lote, detector.detectar_lote(rutasl)):
            detecciones[fr["image_id"]] = dets
        if (i // lote_max) % 50 == 0:
            print(f"  {i + len(lote)}/{len(frames)} frames | {time.time()-t0:.0f}s", flush=True)
    duracion = time.time() - t0

    preds, descartes = a_predicciones_coco(detecciones, COCO_A_L1)
    DIR_PREDICCIONES.mkdir(parents=True, exist_ok=True)
    ruta_preds = DIR_PREDICCIONES / f"{dataset}-{modelo_id}.json"
    with open(ruta_preds, "w", encoding="utf-8") as f:
        json.dump(preds, f)

    metricas, resumen_txt = evaluar(ruta_gt, preds)
    params = {
        "modelo": modelo_id,
        "candidato": mc["candidato"],
        "variante": mc["variante"],
        "dataset": dataset,
        "imgsz": cfg["imgsz"],
        "conf_min": cfg["conf_min"],
        "max_dets": 100,
        "semilla": cfg["semilla"],
        "manifest_sha256": sha256_fichero(DERIVADOS / f"manifest-{dataset}.json"),
        "gt_sha256": sha256_fichero(ruta_gt),
        "predicciones_sha256": sha256_fichero(ruta_preds),
        "n_frames": len(frames),
        "n_gt_person": manifest["resumen"]["gt_person"],
        "n_gt_vehicle": manifest["resumen"]["gt_vehicle"],
        "duracion_inferencia_s": round(duracion, 1),
        **info_pesos(mc),
        **params_comunes(),
    }
    tags = {"licencia": mc["licencia"], "runtime": "pytorch", "fase": "linea-base-zero-shot"}
    registrar_corrida(
        f"eval-{dataset}-{modelo_id}",
        params,
        metricas,
        tags,
        {"cocoeval-resumen.txt": resumen_txt, "descartes-clases.json": json.dumps(dict(descartes))},
    )
    print(f"== eval {modelo_id} / {dataset} ({duracion:.0f}s inferencia) ==")
    for k in ("mAP50", "mAP50_95", "AP50_person", "AP50_vehicle", "AP_small", "AP_medium", "AP_large"):
        print(f"  {k}: {metricas[k]:.4f}")
    print(f"  descartes de clases COCO fuera de L1: {dict(descartes)}")
    print(resumen_txt)


# ---------- exportar / latencia ----------

def exportar_modelo(modelo_id, cfg, bloquear_gridsample=False):
    from harness import latencia as lat

    mc = config_modelo(cfg, modelo_id)
    if mc["candidato"] == "yolo-ultralytics":
        ruta = lat.exportar_yolo_onnx(rutas.PROVIDENCE_DATA / mc["pesos"], DIR_ONNX, cfg["imgsz"])
    else:
        ruta, _ = lat.exportar_hf_onnx(
            mc["hf_id"], DIR_ONNX, modelo_id, cfg["imgsz"], bloquear_gridsample
        )
    print(f"exportado: {ruta}")
    return ruta


def frames_latencia(cfg):
    seleccion = []
    for dataset in ("virat", "meva"):
        with open(DERIVADOS / f"manifest-{dataset}.json", encoding="utf-8") as f:
            frames = json.load(f)["frames"]
        paso = max(1, len(frames) // cfg["latencia"]["frames_por_dataset"])
        seleccion += [DIR_FRAMES / fr["file_name"] for fr in frames[::paso]][
            : cfg["latencia"]["frames_por_dataset"]
        ]
    return seleccion


def latencia_modelo(modelo_id, cfg):
    import cv2

    from harness import latencia as lat
    from harness.registro_mlflow import params_comunes, registrar_corrida, sha256_fichero

    mc = config_modelo(cfg, modelo_id)
    es_yolo = mc["candidato"] == "yolo-ultralytics"
    nombre_onnx = (
        Path(mc["pesos"]).stem + ".onnx" if es_yolo else f"{modelo_id}.onnx"
    )
    ruta_onnx = DIR_ONNX / nombre_onnx
    sesion = lat.cargar_sesion(ruta_onnx)
    dtype = lat.dtype_de_entrada(sesion)

    processor = None
    if not es_yolo:
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(mc["hf_id"], local_files_only=True)

    rutas_frames = frames_latencia(cfg)
    entradas = []
    t0 = time.time()
    for ruta in rutas_frames:
        bgr = cv2.imread(str(ruta))
        if es_yolo:
            x, _, _ = lat.preprocesar_yolo(bgr, cfg["imgsz"], dtype)
        else:
            x = lat.preprocesar_hf(processor, bgr, dtype)
        entradas.append(x)
    ms_prepro = (time.time() - t0) * 1000.0 / len(entradas)

    est = lat.medir(sesion, entradas, cfg["latencia"]["warmup"], cfg["latencia"]["medidas"])
    est["pasa_15fps"] = 1.0 if est["fps_media"] >= cfg["latencia"]["umbral_fps"] else 0.0
    est["ms_preprocesado_ref"] = round(ms_prepro, 2)
    est.update(sanity_fp16(sesion, mc, cfg, rutas_frames[:20], dtype, processor))

    import torch

    params = {
        "modelo": modelo_id,
        "candidato": mc["candidato"],
        "variante": mc["variante"],
        "onnx": str(ruta_onnx),
        "onnx_sha256": sha256_fichero(ruta_onnx),
        "dtype_entrada": np_dtype_nombre(dtype),
        "provider": sesion.get_providers()[0],
        "gpu": torch.cuda.get_device_name(0),
        "imgsz": cfg["imgsz"],
        "warmup": cfg["latencia"]["warmup"],
        "n_frames_reales": len(entradas),
        **info_pesos(mc),
        **params_comunes(),
    }
    tags = {"licencia": mc["licencia"], "runtime": "onnxruntime-gpu", "fase": "linea-base-zero-shot"}
    registrar_corrida(f"lat-{modelo_id}", params, est, tags)
    print(
        f"== latencia {modelo_id} == media {est['ms_media']:.2f} ms | p50 {est['ms_p50']:.2f} | "
        f"p95 {est['ms_p95']:.2f} | {est['fps_media']:.1f} FPS | pasa>=15FPS: {bool(est['pasa_15fps'])} "
        f"| prepro(ref): {est['ms_preprocesado_ref']:.2f} ms | provider: {params['provider']}"
    )
    print(
        f"   sanity FP16 (conf>=0.3, {est['sanity_n_frames']} frames): ref {est['sanity_ref_total']} "
        f"dets, onnx {est['sanity_onnx_total']}, emparejadas {est['sanity_emparejadas']}, "
        f"IoU media {est['sanity_iou_media']:.4f}, delta conf media {est['sanity_delta_conf_media']:.4f}"
    )
    return est


def np_dtype_nombre(dtype):
    import numpy as np

    return np.dtype(dtype).name


def sanity_fp16(sesion, mc, cfg, rutas_frames, dtype, processor, conf=0.3):
    """Advertencia 7: desviación ONNX-FP16 vs PyTorch-FP32 en ~20 frames reales."""
    import cv2
    import numpy as np

    from harness import latencia as lat

    nombre_entrada = sesion.get_inputs()[0].name
    es_yolo = mc["candidato"] == "yolo-ultralytics"
    comparaciones = []

    if es_yolo:
        from ultralytics import YOLO

        ref = YOLO(str(rutas.PROVIDENCE_DATA / mc["pesos"]))
        for ruta in rutas_frames:
            bgr = cv2.imread(str(ruta))
            r = ref.predict(
                str(ruta), imgsz=cfg["imgsz"], conf=conf, max_det=300, device=0, verbose=False
            )[0]
            dets_ref = [
                (int(c), float(cf), *(float(v) for v in xy))
                for xy, cf, c in zip(
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                )
            ]
            x, escala, despl = lat.preprocesar_yolo(bgr, cfg["imgsz"], dtype)
            salida = sesion.run(None, {nombre_entrada: x})
            dets_onnx = lat.decodificar_yolo_onnx(salida, escala, despl, conf)
            comparaciones.append(lat.comparar_detecciones(dets_ref, dets_onnx))
    else:
        from harness.inferencia.hf_detr import DetectorHF

        ref = DetectorHF(mc["hf_id"], conf=conf)
        for ruta in rutas_frames:
            bgr = cv2.imread(str(ruta))
            dets_ref = [
                (nombre, cf, x, y, x + w, y + h)
                for nombre, cf, x, y, w, h in ref.detectar_lote([ruta])[0]
            ]
            x = lat.preprocesar_hf(processor, bgr, dtype)
            salida = sesion.run(None, {nombre_entrada: x})
            dets_onnx = [
                (ref.id2label[int(c)], cf, *caja)
                for c, cf, *caja in lat.decodificar_hf_onnx(salida, bgr.shape[0], bgr.shape[1], conf)
            ]
            comparaciones.append(lat.comparar_detecciones(dets_ref, dets_onnx))

    return {
        "sanity_n_frames": len(comparaciones),
        "sanity_ref_total": sum(c["n_ref"] for c in comparaciones),
        "sanity_onnx_total": sum(c["n_onnx"] for c in comparaciones),
        "sanity_emparejadas": sum(c["emparejadas"] for c in comparaciones),
        "sanity_iou_media": float(np.nanmean([c["iou_media"] for c in comparaciones])),
        "sanity_delta_conf_media": float(np.nanmean([c["delta_conf_media"] for c in comparaciones])),
    }


# ---------- análisis tipo 0 VIRAT ----------

def analizar_tipo0():
    conteo = {}
    for anot in sorted(rutas.VIRAT_ANOTACIONES.glob("*.objects.txt")):
        c = Counter()
        with open(anot, encoding="utf-8") as f:
            for linea in f:
                partes = linea.split()
                if len(partes) == 8:
                    c[partes[7]] += 1
        if c.get("0"):
            conteo[anot.name.split(".")[0]] = {"tipo0": c["0"], "total": sum(c.values())}
    total0 = sum(v["tipo0"] for v in conteo.values())
    total = sum(v["total"] for v in conteo.values())
    doc = {"videos_con_tipo0": len(conteo), "filas_tipo0": total0, "por_video": conteo}
    with open(DERIVADOS / "analisis-tipo0-virat.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    print(f"tipo 0: {total0} filas en {len(conteo)} vídeos (sobre filas de vídeos afectados: {total})")
    top = sorted(conteo.items(), key=lambda kv: -kv[1]["tipo0"])[:10]
    for nombre, v in top:
        print(f"  {nombre}: {v['tipo0']} filas tipo 0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preparar")
    p.add_argument("--dataset", choices=["virat", "meva"], required=True)
    p = sub.add_parser("evaluar")
    p.add_argument("--modelo", required=True)
    p.add_argument("--dataset", choices=["virat", "meva"], required=True)
    p = sub.add_parser("exportar")
    p.add_argument("--modelo", required=True)
    p.add_argument("--bloquear-gridsample", action="store_true")
    p = sub.add_parser("latencia")
    p.add_argument("--modelo", required=True)
    sub.add_parser("tipo0-virat")
    args = parser.parse_args()

    cfg = cargar_config()
    if args.cmd == "preparar":
        preparar_dataset(args.dataset, cfg)
    elif args.cmd == "evaluar":
        evaluar_modelo(args.modelo, args.dataset, cfg)
    elif args.cmd == "exportar":
        exportar_modelo(args.modelo, cfg, args.bloquear_gridsample)
    elif args.cmd == "latencia":
        latencia_modelo(args.modelo, cfg)
    elif args.cmd == "tipo0-virat":
        analizar_tipo0()


if __name__ == "__main__":
    main()
