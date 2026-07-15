"""Muestreo determinista de frames de evaluación (protocolo aprobado en B3).

Elegibles = frames con >=1 caja GT tras el mapeo L1, intersecados con el rango
decodificable del vídeo (neutraliza el off-by-one de VIRAT, aprobado 2026-07-16).
Selección: barrido greedy con zancada temporal mínima; si supera el tope, se
adelgaza por espaciado uniforme sobre la lista seleccionada. Sin azar.
"""


def seleccionar_frames(frames_elegibles, n_frames_video, zancada_min, tope):
    """Devuelve la lista ordenada de frames seleccionados."""
    validos = [f for f in sorted(frames_elegibles) if 0 <= f < n_frames_video]
    if not validos:
        return []

    escogidos = []
    ultimo = None
    for frame in validos:
        if ultimo is None or frame >= ultimo + zancada_min:
            escogidos.append(frame)
            ultimo = frame

    if len(escogidos) <= tope:
        return escogidos
    # Adelgazado uniforme y determinista hasta el tope.
    paso = (len(escogidos) - 1) / (tope - 1) if tope > 1 else 0
    indices = sorted({round(i * paso) for i in range(tope)})
    return [escogidos[i] for i in indices]
