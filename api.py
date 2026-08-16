# -----------------------------------------------------------------------------------------
# API REST - Lector de senas (vocales)
# -----------------------------------------------------------------------------------------
# Expone la logica de inferencia (inferencia.py) como un servicio HTTP con FastAPI.
#
# Para correrla localmente:
#   uvicorn api:app --reload
#
# Luego se puede probar en el navegador la documentacion interactiva en:
#   http://127.0.0.1:8000/docs
# -----------------------------------------------------------------------------------------

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from inferencia import CLASES, predecir_desde_frame

app = FastAPI(
    title="API Lector de Senas (vocales)",
    description=(
        "Recibe una imagen con una mano haciendo una sena y devuelve la vocal "
        "reconocida (A, E, I, O, U), si es que hay una mano con confianza suficiente."
    ),
    version="1.0.0",
)


@app.get("/salud")
def salud():
    """Chequeo simple de que la API esta arriba y el modelo quedo cargado."""
    return {"estado": "ok"}


@app.get("/clases")
def clases():
    """Devuelve las clases que el modelo puede reconocer, en el orden que usa internamente."""
    return {"clases": CLASES}


@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    """
    Recibe una imagen (foto o frame de camara) como archivo (multipart/form-data) y
    devuelve la vocal detectada.

    Respuesta:
      - mano_detectada: si se encontro una mano en la imagen.
      - letra: la vocal reconocida, o null si no hubo mano o la confianza fue baja.
      - confianza: probabilidad (0 a 1) de la clase elegida, o null.
      - caja: recuadro (x1, y1, x2, y2) de la mano detectada, o null.
    """
    contenido = await imagen.read()
    datos = np.frombuffer(contenido, dtype=np.uint8)
    frame = cv2.imdecode(datos, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")

    return predecir_desde_frame(frame)
