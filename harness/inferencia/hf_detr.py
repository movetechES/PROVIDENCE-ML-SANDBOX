"""Candidatos Apache-2.0: RT-DETRv2 y D-FINE vía transformers (un solo wrapper).

Los pesos se leen SOLO de la caché local (HF_HOME apuntando a
PROVIDENCE-DATA\\pesos\\linea-base\\hf-cache; las descargas las ejecuta
Alejandro): local_files_only=True impide cualquier descarga desde aquí.
Las imágenes se leen con cv2 (BGR->RGB); no se usa PIL (no es dependencia
aprobada). El preprocesado es el canónico del processor de cada modelo.
"""

import cv2
import torch


class DetectorHF:
    def __init__(self, hf_id, conf=0.001, device="cuda"):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.processor = AutoImageProcessor.from_pretrained(hf_id, local_files_only=True)
        self.modelo = AutoModelForObjectDetection.from_pretrained(hf_id, local_files_only=True)
        self.modelo.eval().to(device)
        self.id2label = self.modelo.config.id2label
        self.conf = conf
        self.device = device

    def detectar_lote(self, rutas):
        imagenes, tamanos = [], []
        for ruta in rutas:
            bgr = cv2.imread(str(ruta))
            if bgr is None:
                raise RuntimeError(f"No se pudo leer {ruta}")
            imagenes.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            tamanos.append((bgr.shape[0], bgr.shape[1]))  # (alto, ancho)

        entradas = self.processor(images=imagenes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            salidas = self.modelo(**entradas)
        procesadas = self.processor.post_process_object_detection(
            salidas, threshold=self.conf, target_sizes=torch.tensor(tamanos, device=self.device)
        )

        lote = []
        for res in procesadas:
            detecciones = []
            for score, label, box in zip(
                res["scores"].cpu().numpy(),
                res["labels"].cpu().numpy(),
                res["boxes"].cpu().numpy(),
            ):
                x1, y1, x2, y2 = (float(v) for v in box)
                detecciones.append(
                    (self.id2label[int(label)], float(score), x1, y1, x2 - x1, y2 - y1)
                )
            lote.append(detecciones)
        return lote
