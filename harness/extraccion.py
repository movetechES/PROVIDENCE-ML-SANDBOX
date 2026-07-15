"""Extracción de frames de evaluación: decodificación secuencial exacta.

Se usa grab() secuencial (sin seeks: los índices de frame de las anotaciones
deben corresponder exactamente al frame decodificado) y retrieve() solo en los
frames muestreados. JPEG calidad 95 (decisión 4 de la aprobación de B3).
"""

import cv2


def extraer_frames(ruta_video, indices, dir_salida, prefijo, jpeg_calidad=95):
    """Extrae los frames `indices` (ordenados) de un vídeo.

    Devuelve (extraidos, fallidos): listas de (indice, nombre_fichero) y de índices
    que no se pudieron decodificar (se excluyen del manifest y del GT).
    """
    dir_salida.mkdir(parents=True, exist_ok=True)
    pendientes = sorted(set(indices))
    extraidos, fallidos = [], []
    if not pendientes:
        return extraidos, fallidos

    captura = cv2.VideoCapture(str(ruta_video))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir {ruta_video}")
    objetivo = set(pendientes)
    ultimo = pendientes[-1]
    frame_actual = -1
    try:
        while frame_actual < ultimo:
            if not captura.grab():
                break
            frame_actual += 1
            if frame_actual not in objetivo:
                continue
            ok, imagen = captura.retrieve()
            nombre = f"{prefijo}_f{frame_actual:06d}.jpg"
            if not ok:
                fallidos.append(frame_actual)
                continue
            destino = dir_salida / nombre
            if not cv2.imwrite(str(destino), imagen, [cv2.IMWRITE_JPEG_QUALITY, jpeg_calidad]):
                raise RuntimeError(f"No se pudo escribir {destino}")
            extraidos.append((frame_actual, nombre))
    finally:
        captura.release()

    decodificados = {f for f, _ in extraidos}
    fallidos.extend(f for f in pendientes if f not in decodificados and f not in fallidos)
    return extraidos, sorted(set(fallidos))


def contar_frames(ruta_video):
    """(n_frames, ancho, alto) según los metadatos del contenedor."""
    captura = cv2.VideoCapture(str(ruta_video))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir {ruta_video}")
    n = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    captura.release()
    return n, ancho, alto
