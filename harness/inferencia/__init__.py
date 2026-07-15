"""Wrappers de inferencia de los candidatos.

Contrato común: detectar_lote(rutas_imagenes) -> list[list[(nombre_clase, conf,
x, y, w, h)]] en píxeles de la imagen original. El mapeo a L1 (por NOMBRE de
clase) lo hace el evaluador, no el wrapper.
"""
