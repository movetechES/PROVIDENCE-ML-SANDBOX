"""Candidato AGPL: YOLO vía ultralytics. El import de ultralytics vive SOLO en
este módulo (aislamiento del candidato AGPL; CLAUDE.md, frontera de licencias).
"""


class DetectorYolo:
    def __init__(self, ruta_pesos, imgsz=640, conf=0.001, device=0):
        from ultralytics import YOLO

        self.modelo = YOLO(str(ruta_pesos))
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

    def detectar_lote(self, rutas):
        resultados = self.modelo.predict(
            [str(r) for r in rutas],
            imgsz=self.imgsz,
            conf=self.conf,
            max_det=300,
            device=self.device,
            verbose=False,
        )
        salida = []
        for res in resultados:
            detecciones = []
            nombres = res.names
            cajas = res.boxes
            for xyxy, conf, cls in zip(
                cajas.xyxy.cpu().numpy(), cajas.conf.cpu().numpy(), cajas.cls.cpu().numpy()
            ):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                detecciones.append(
                    (nombres[int(cls)], float(conf), x1, y1, x2 - x1, y2 - y1)
                )
            salida.append(detecciones)
        return salida
