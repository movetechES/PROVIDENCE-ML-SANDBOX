"""Registro MLflow LOCAL (mlruns/ del sandbox, gitignorado) — plan_ml §0.6.

Cada corrida registra config, semilla, versión DVC de los datos, hashes de
pesos/manifests y versiones de librerías (reproducibilidad).
"""

import hashlib
import subprocess
from pathlib import Path

import mlflow
import yaml

from harness import rutas

RAIZ_SANDBOX = Path(__file__).resolve().parents[1]
EXPERIMENTO = "linea-base-zero-shot"


def sha256_fichero(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        while trozo := f.read(bloque):
            h.update(trozo)
    return h.hexdigest()


def md5_dvc(nombre):
    """md5 DVC del dataset externo ('virat' | 'meva' | 'meva-anotaciones')."""
    ruta = rutas.PROVIDENCE_DATA / "externos" / f"{nombre}.dvc"
    with open(ruta, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    return doc["outs"][0]["md5"]


def commit_git_sandbox():
    res = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=RAIZ_SANDBOX
    )
    return res.stdout.strip() or "desconocido"


def versiones():
    import cv2
    import numpy
    import onnxruntime
    import torch
    import transformers

    v = {
        "v_torch": torch.__version__,
        "v_transformers": transformers.__version__,
        "v_onnxruntime": onnxruntime.__version__,
        "v_opencv": cv2.__version__,
        "v_numpy": numpy.__version__,
        "v_mlflow": mlflow.__version__,
        "v_pyyaml": yaml.__version__,
    }
    try:
        import ultralytics

        v["v_ultralytics"] = ultralytics.__version__
    except ImportError:
        pass
    try:
        import pycocotools

        v["v_pycocotools"] = getattr(pycocotools, "__version__", "2.x")
    except ImportError:
        pass
    return v


def params_comunes():
    return {
        "git_commit_sandbox": commit_git_sandbox(),
        "dvc_md5_virat": md5_dvc("virat"),
        "dvc_md5_meva": md5_dvc("meva"),
        "dvc_md5_meva_anotaciones": md5_dvc("meva-anotaciones"),
        **versiones(),
    }


def registrar_corrida(nombre, params, metricas, tags, artefactos_texto=None):
    """artefactos_texto: dict nombre_fichero -> contenido (p. ej. resumen COCOeval).

    Backend sqlite local (mlflow.db, gitignorado): el file store de MLflow 3.x
    está en modo mantenimiento y rechaza escrituras por defecto.
    """
    mlflow.set_tracking_uri("sqlite:///" + (RAIZ_SANDBOX / "mlflow.db").as_posix())
    mlflow.set_experiment(EXPERIMENTO)
    with mlflow.start_run(run_name=nombre):
        mlflow.log_params(params)
        metricas_validas = {
            k: v for k, v in metricas.items() if isinstance(v, (int, float)) and v == v
        }
        mlflow.log_metrics(metricas_validas)
        mlflow.set_tags(tags)
        for fichero, contenido in (artefactos_texto or {}).items():
            mlflow.log_text(contenido, fichero)
