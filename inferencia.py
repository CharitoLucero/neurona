# -----------------------------------------------------------------------------------------
# MODULO DE INFERENCIA
# -----------------------------------------------------------------------------------------
# Separa la deteccion de mano y la prediccion de vocal del bucle de camara, para poder
# reutilizar la misma logica tanto desde el script de escritorio (prediccion.py) como
# desde la API (api.py). Al importar este modulo, el modelo se carga UNA sola vez.
# -----------------------------------------------------------------------------------------

import os

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from recorte import LADO_SALIDA, recortar_mano

# --- Rutas del modelo entrenado (relativas a este archivo, para que funcione sin importar
#     desde donde se ejecute el proceso) ---
_RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
# modeloVocales_mano.keras se entreno con entrenamiento.py sobre recortes generados
# con recorte.recortar_mano, es decir EXACTAMENTE el mismo encuadre que se usa aca abajo al
# predecir. Que el entrenamiento y la prediccion vean la misma cosa es lo unico que hace que
# el modelo funcione en vivo (ver DOCUMENTACION.md, secciones 12 y 15).
RUTA_MODELO = os.path.join(_RUTA_BASE, "modeloVocales_mano.keras")
RUTA_PESOS = os.path.join(_RUTA_BASE, "pesosVocales_mano.weights.h5")

# Tamaño de imagen que espera el modelo (lo define recorte.py, que es lo que arma el recorte).
ALTURA = LONGITUD = LADO_SALIDA

# Orden de las clases: Keras (flow_from_directory) asigna los indices ordenando las
# subcarpetas del dataset alfabeticamente. Mientras el dataset se mantenga con las
# subcarpetas A/E/I/O/U, este orden es el mismo que usó el entrenamiento. Si se agregan
# o reordenan clases, hay que actualizar esta lista.
CLASES = ["A", "E", "I", "O", "U"]

# Por debajo de esta confianza, se considera que no hay una seña reconocible.
UMBRAL_CONFIANZA = 0.6

# --- Carga del modelo (una sola vez, al importar el modulo) ---
# compile=False: el archivo .keras guarda tambien el estado del optimizador (Adam),
# que quedo en un formato HDF5 que esta version de Keras no puede leer ("bad object
# header version number"). Para inferencia no se necesita el optimizador (solo se
# usa para reentrenar), asi que se omite y se cargan directamente los pesos de las capas.
_modelo = load_model(RUTA_MODELO, compile=False)
_modelo.load_weights(RUTA_PESOS)

# --- Detector de manos de MediaPipe (una sola instancia reutilizada entre llamadas) ---
_manos_mp = mp.solutions.hands
_detector_manos = _manos_mp.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)


def _preprocesar(recorte_bgr):
    """
    Prepara el recorte para la red: lo pasa a RGB y reescala los pixeles a [0,1].
    (El redimensionado a 200x200 ya lo hizo recortar_mano.)

    El /255 replica el rescale=1./255 de ImageDataGenerator que se usa al entrenar; si
    faltara, el modelo recibiria valores 0-255 y no se pareceria a lo que vio entrenando.
    """
    recorte_rgb = cv2.cvtColor(recorte_bgr, cv2.COLOR_BGR2RGB)
    x = img_to_array(recorte_rgb) / 255.0
    return np.expand_dims(x, axis=0)


def predecir_desde_frame(frame_bgr):
    """
    Detecta una mano en el frame (formato BGR, como lo entrega OpenCV) y predice la vocal.

    Devuelve un diccionario con:
      - mano_detectada: bool
      - letra: str o None (None si no hay mano o la confianza es baja)
      - confianza: float o None
      - caja: (x1, y1, x2, y2) o None
      - probabilidades: dict {letra: probabilidad} de las 5 clases, o None si no hubo mano/recorte
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resultado_mp = _detector_manos.process(frame_rgb)

    if not resultado_mp.multi_hand_landmarks:
        return {"mano_detectada": False, "letra": None, "confianza": None, "caja": None, "probabilidades": None}

    landmarks = resultado_mp.multi_hand_landmarks[0].landmark
    recorte, caja = recortar_mano(frame_bgr, landmarks)

    if recorte is None or recorte.size == 0:
        return {"mano_detectada": True, "letra": None, "confianza": None, "caja": caja, "probabilidades": None}

    x = _preprocesar(recorte)
    probabilidades = _modelo.predict(x, verbose=0)[0]
    indice = int(np.argmax(probabilidades))
    confianza = float(probabilidades[indice])
    # Probabilidad de cada clase, util para depurar por que el modelo elige (o descarta) una letra.
    dist_probabilidades = {CLASES[i]: round(float(p), 3) for i, p in enumerate(probabilidades)}

    if confianza < UMBRAL_CONFIANZA:
        return {"mano_detectada": True, "letra": None, "confianza": confianza, "caja": caja, "probabilidades": dist_probabilidades}

    return {"mano_detectada": True, "letra": CLASES[indice], "confianza": confianza, "caja": caja, "probabilidades": dist_probabilidades}
