"""Conversor MEVA KPF: geom.yml + types.yml -> cajas L1 por frame.

Resolución de fuente por clip en el orden aprobado en B3: kitware/ primero,
kitware-meva-training/ después. KPF: cada entrada 'geom' trae g0 = "x1 y1 x2 y2"
en píxeles y ts0 = índice de frame; el tipo del track (id1) se resuelve contra
types.yml (clave de mayor confianza del cset3). Los clips «marked empty»
devuelven cero frames. Se incluyen las cajas interpoladas (keyframe: false):
son parte de la anotación oficial.
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field

import yaml

from harness import rutas
from harness.conversores.comun import ConstructorCocoGT, recortar_caja
from harness.ontologia import MEVA_A_L1

# CSafeLoader (libyaml) aprobado en REGISTRO.md 2026-07-16: los geom.yml grandes
# rondan las 100k líneas y el loader puro de Python los hace inviables.
LOADER = yaml.CSafeLoader


@dataclass
class ResultadoMeva:
    clip: str
    fuente: str
    ancho: int
    alto: int
    marcado_vacio: bool = False
    cajas_por_frame: dict = field(default_factory=dict)  # frame -> [(cat, x, y, w, h)]
    tracks_por_etiqueta: Counter = field(default_factory=Counter)
    geoms_totales: int = 0
    geoms_sin_tipo: int = 0
    cajas_descartadas_por_etiqueta: Counter = field(default_factory=Counter)
    cajas_degeneradas: int = 0
    cajas_recortadas: int = 0

    @property
    def cajas_mapeadas(self):
        return sum(len(c) for c in self.cajas_por_frame.values())


def localizar_anotacion(clip, raiz=None):
    """Devuelve (fuente, ruta al geom.yml) o (None, None). El fichero puede no
    colgar de la carpeta-hora de inicio del clip, de ahí el glob por fecha."""
    raiz = raiz or rutas.MEVA_ANOTACIONES
    fecha = clip.split(".")[0]
    for fuente in rutas.MEVA_FUENTES:
        base = raiz / fuente / fecha
        if not base.is_dir():
            continue
        candidatos = sorted(base.glob(f"*/{clip}.geom.yml"))
        if candidatos:
            return fuente, candidatos[0]
    return None, None


def _cargar_kpf(ruta):
    with open(ruta, encoding="utf-8") as fichero:
        return yaml.load(fichero, Loader=LOADER) or []


def _indice_tipos(ruta_types):
    """id1 -> etiqueta cset3 de mayor confianza."""
    tipos = {}
    for entrada in _cargar_kpf(ruta_types):
        registro = entrada.get("types") if isinstance(entrada, dict) else None
        if not registro:
            continue
        cset = registro.get("cset3") or {}
        if cset:
            tipos[registro["id1"]] = max(cset, key=cset.get)
    return tipos


def convertir_clip(clip, ancho, alto, raiz=None):
    """Parsea el KPF de un clip y devuelve las cajas L1 por frame."""
    fuente, ruta_geom = localizar_anotacion(clip, raiz)
    if ruta_geom is None:
        raise FileNotFoundError(f"Sin anotación KPF para {clip} en {rutas.MEVA_FUENTES}")
    ruta_types = ruta_geom.with_name(ruta_geom.name.replace(".geom.yml", ".types.yml"))

    tipos = _indice_tipos(ruta_types)
    res = ResultadoMeva(clip=clip, fuente=fuente, ancho=ancho, alto=alto)
    res.tracks_por_etiqueta = Counter(tipos.values())

    for entrada in _cargar_kpf(ruta_geom):
        if not isinstance(entrada, dict):
            continue
        meta = entrada.get("meta")
        if isinstance(meta, str) and "marked empty" in meta:
            res.marcado_vacio = True
            continue
        geom = entrada.get("geom")
        if not geom:
            continue
        res.geoms_totales += 1

        etiqueta = tipos.get(geom["id1"])
        if etiqueta is None:
            res.geoms_sin_tipo += 1
            continue
        categoria = MEVA_A_L1.get(etiqueta)
        if categoria is None:
            res.cajas_descartadas_por_etiqueta[etiqueta] += 1
            continue

        x1, y1, x2, y2 = (float(v) for v in str(geom["g0"]).split())
        caja, recortada = recortar_caja(x1, y1, x2, y2, ancho, alto)
        if caja is None:
            res.cajas_degeneradas += 1
            continue
        if recortada:
            res.cajas_recortadas += 1
        res.cajas_por_frame.setdefault(int(geom["ts0"]), []).append((categoria, *caja))
    return res


def main():
    import cv2

    parser = argparse.ArgumentParser(description="Smoke test del conversor MEVA KPF")
    parser.add_argument("clip")
    parser.add_argument("--frames-muestra", type=int, default=2)
    args = parser.parse_args()

    fecha = args.clip.split(".")[0]
    videos = sorted((rutas.MEVA_VIDEOS / fecha).rglob(f"{args.clip}*.avi"))
    if not videos:
        raise FileNotFoundError(f"Sin video para {args.clip}")
    captura = cv2.VideoCapture(str(videos[0]))
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_video = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    captura.release()

    res = convertir_clip(args.clip, ancho, alto)

    print(f"== MEVA {args.clip} ==")
    print(f"video: {videos[0].name}, {ancho}x{alto}, {n_frames_video} frames")
    print(f"fuente de anotación: {res.fuente} | marcado vacío: {res.marcado_vacio}")
    print(f"tracks por etiqueta (types.yml): {dict(res.tracks_por_etiqueta) or '{}'}")
    print(
        f"geoms: {res.geoms_totales} | sin tipo: {res.geoms_sin_tipo} | "
        f"descartadas por etiqueta: {dict(res.cajas_descartadas_por_etiqueta) or '{}'}"
    )
    print(
        f"cajas mapeadas: {res.cajas_mapeadas} | degeneradas: {res.cajas_degeneradas} "
        f"| recortadas al marco: {res.cajas_recortadas}"
    )
    frames = sorted(res.cajas_por_frame)
    if frames:
        print(f"frames con >=1 GT: {len(frames)} (rango {frames[0]}..{frames[-1]})")
    else:
        print("frames con >=1 GT: 0")

    constructor = ConstructorCocoGT()
    for frame in frames[: args.frames_muestra]:
        constructor.agregar_frame(
            f"{args.clip}_f{frame:06d}.jpg", ancho, alto, res.cajas_por_frame[frame]
        )
    print(f"-- GT COCO de muestra ({min(args.frames_muestra, len(frames))} frames) --")
    print(json.dumps(constructor.coco, indent=1))


if __name__ == "__main__":
    main()
