"""Rutas de datos del benchmark. Los datos viven FUERA del repo (CLAUDE.md, regla a)."""

from pathlib import Path

PROVIDENCE_DATA = Path(r"C:\DEV-HEROS-DEFENSE\PROVIDENCE-DATA")

VIRAT_VIDEOS = PROVIDENCE_DATA / "externos" / "virat" / "videos_original"
VIRAT_ANOTACIONES = PROVIDENCE_DATA / "externos" / "virat" / "annotations"

MEVA_VIDEOS = PROVIDENCE_DATA / "externos" / "meva" / "video"
MEVA_SUBCONJUNTO = PROVIDENCE_DATA / "externos" / "meva" / "subconjunto-clips.txt"
MEVA_ANOTACIONES = (
    PROVIDENCE_DATA / "externos" / "meva-anotaciones" / "annotation" / "DIVA-phase-2" / "MEVA"
)
# Orden de prioridad aprobado en B3: kitware primero, luego kitware-meva-training.
MEVA_FUENTES = ("kitware", "kitware-meva-training")

PESOS = PROVIDENCE_DATA / "pesos" / "linea-base"
DERIVADOS = PROVIDENCE_DATA / "derivados" / "linea-base-frames-v1"
