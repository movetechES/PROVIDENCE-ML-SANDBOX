"""Ontología Nivel 1 del benchmark y mapeos por dataset (aprobados en B3, fase 1).

Los descartes son simétricos entre GT y predicciones: bicis/motos no puntúan
ni a favor ni en contra (advertencia 3 de la propuesta aprobada).
"""

PERSON = 1
VEHICLE = 2

CATEGORIAS = [
    {"id": PERSON, "name": "person"},
    {"id": VEHICLE, "name": "vehicle"},
]

# VIRAT objects.txt, columna 8 (docs/README_format_release2.txt).
# 4 (object, objetos portados) y 5 (bike/bicycles, incl. motos) se descartan.
VIRAT_A_L1 = {1: PERSON, 2: VEHICLE, 3: VEHICLE}
VIRAT_TIPOS = {1: "person", 2: "car", 3: "vehicles", 4: "object", 5: "bike"}

# MEVA KPF types.yml (cset3). 'other' (cajón de sastre ActEV) y 'bag' se descartan.
MEVA_A_L1 = {"person": PERSON, "vehicle": VEHICLE}

# Predicciones COCO-80 de los pesos zero-shot, mapeadas por NOMBRE de clase
# (model.names / id2label), nunca por índice. bicycle/motorcycle descartadas
# (simétrico con el GT); el resto de las 80 clases, fuera de la ontología.
COCO_A_L1 = {"person": PERSON, "car": VEHICLE, "bus": VEHICLE, "truck": VEHICLE}
