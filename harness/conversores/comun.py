"""Ensamblado del GT COCO común del benchmark y recorte de cajas al marco.

Las cajas se recortan al marco de la imagen (convención COCO); las que quedan
degeneradas (ancho o alto nulo) se descartan y se cuentan aparte.
"""

from harness.ontologia import CATEGORIAS


def recortar_caja(x1, y1, x2, y2, ancho, alto):
    """Recorta [x1,y1,x2,y2] al marco. Devuelve (caja_xywh | None, fue_recortada)."""
    x1r, y1r = max(0.0, x1), max(0.0, y1)
    x2r, y2r = min(float(ancho), x2), min(float(alto), y2)
    if x2r <= x1r or y2r <= y1r:
        return None, False
    recortada = (x1r, y1r, x2r, y2r) != (x1, y1, x2, y2)
    return (x1r, y1r, x2r - x1r, y2r - y1r), recortada


class ConstructorCocoGT:
    """Acumula frames y cajas L1 y produce el dict COCO del GT."""

    def __init__(self):
        self.coco = {
            "images": [],
            "annotations": [],
            "categories": [dict(c) for c in CATEGORIAS],
        }

    def agregar_frame(self, nombre_fichero, ancho, alto, cajas):
        """cajas: iterable de (categoria_id, x, y, w, h). Devuelve el image_id."""
        image_id = len(self.coco["images"]) + 1
        self.coco["images"].append(
            {"id": image_id, "file_name": nombre_fichero, "width": ancho, "height": alto}
        )
        for categoria_id, x, y, w, h in cajas:
            self.coco["annotations"].append(
                {
                    "id": len(self.coco["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": categoria_id,
                    "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                }
            )
        return image_id
