"""Conversor VIRAT release 2.0: objects.txt -> cajas L1 por frame.

Formato (docs/README_format_release2.txt), 8 columnas por línea:
  object_id, duración, frame, bbox_x, bbox_y, bbox_w, bbox_h, tipo
Tipos: 1 person, 2 car, 3 vehicles -> ontología L1; 4 object y 5 bike se
descartan (aprobado en B3; descarte simétrico con las predicciones).
"""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from harness import rutas
from harness.conversores.comun import ConstructorCocoGT, recortar_caja
from harness.ontologia import VIRAT_A_L1, VIRAT_TIPOS


@dataclass
class ResultadoVirat:
    video_id: str
    ancho: int
    alto: int
    cajas_por_frame: dict = field(default_factory=dict)  # frame -> [(cat, x, y, w, h)]
    filas_totales: int = 0
    filas_malformadas: int = 0
    filas_por_tipo: Counter = field(default_factory=Counter)
    tracks_por_tipo: dict = field(default_factory=lambda: defaultdict(set))
    cajas_descartadas_por_tipo: Counter = field(default_factory=Counter)
    cajas_degeneradas: int = 0
    cajas_recortadas: int = 0

    @property
    def cajas_mapeadas(self):
        return sum(len(c) for c in self.cajas_por_frame.values())


def convertir_video(video_id, ancho, alto, dir_anotaciones=None):
    """Parsea el objects.txt de un vídeo y devuelve las cajas L1 por frame."""
    dir_anotaciones = dir_anotaciones or rutas.VIRAT_ANOTACIONES
    ruta = dir_anotaciones / f"{video_id}.viratdata.objects.txt"
    res = ResultadoVirat(video_id=video_id, ancho=ancho, alto=alto)

    with open(ruta, encoding="utf-8") as fichero:
        for linea in fichero:
            partes = linea.split()
            if not partes:
                continue
            if len(partes) != 8:
                res.filas_malformadas += 1
                continue
            oid, _dur, frame, x, y, w, h, tipo = (int(float(p)) for p in partes)
            res.filas_totales += 1
            res.filas_por_tipo[tipo] += 1
            res.tracks_por_tipo[tipo].add(oid)

            categoria = VIRAT_A_L1.get(tipo)
            if categoria is None:
                res.cajas_descartadas_por_tipo[tipo] += 1
                continue
            caja, recortada = recortar_caja(x, y, x + w, y + h, ancho, alto)
            if caja is None:
                res.cajas_degeneradas += 1
                continue
            if recortada:
                res.cajas_recortadas += 1
            res.cajas_por_frame.setdefault(frame, []).append((categoria, *caja))
    return res


def _nombre_tipo(tipo):
    return VIRAT_TIPOS.get(tipo, f"desconocido({tipo})")


def main():
    import cv2

    parser = argparse.ArgumentParser(description="Smoke test del conversor VIRAT")
    parser.add_argument("video_id")
    parser.add_argument("--frames-muestra", type=int, default=2)
    args = parser.parse_args()

    video = rutas.VIRAT_VIDEOS / f"{args.video_id}.mp4"
    captura = cv2.VideoCapture(str(video))
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames_video = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    captura.release()

    res = convertir_video(args.video_id, ancho, alto)

    print(f"== VIRAT {args.video_id} ==")
    print(f"video: {ancho}x{alto}, {n_frames_video} frames")
    print(f"filas: {res.filas_totales} (malformadas: {res.filas_malformadas})")
    for tipo in sorted(res.filas_por_tipo):
        destino = VIRAT_A_L1.get(tipo)
        destino_txt = {1: "person", 2: "vehicle"}.get(destino, "DESCARTADO")
        print(
            f"  tipo {tipo} ({_nombre_tipo(tipo)}): {res.filas_por_tipo[tipo]} filas, "
            f"{len(res.tracks_por_tipo[tipo])} tracks -> {destino_txt}"
        )
    print(
        f"cajas mapeadas: {res.cajas_mapeadas} | degeneradas: {res.cajas_degeneradas} "
        f"| recortadas al marco: {res.cajas_recortadas}"
    )
    frames = sorted(res.cajas_por_frame)
    if frames:
        print(f"frames con >=1 GT: {len(frames)} (rango {frames[0]}..{frames[-1]})")

    constructor = ConstructorCocoGT()
    for frame in frames[: args.frames_muestra]:
        constructor.agregar_frame(
            f"{args.video_id}_f{frame:06d}.jpg", ancho, alto, res.cajas_por_frame[frame]
        )
    print(f"-- GT COCO de muestra ({min(args.frames_muestra, len(frames))} frames) --")
    print(json.dumps(constructor.coco, indent=1))


if __name__ == "__main__":
    main()
