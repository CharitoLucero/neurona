# -----------------------------------------------------------------------------------------
# PREPARAR EL DATASET DE RECORTES
# -----------------------------------------------------------------------------------------
# Toma las fotos crudas (fondo real, tal como salieron de la camara) y genera RecortesCrudos/,
# aplicandoles EL MISMO recorte que usa la prediccion en vivo (recorte.recortar_mano). Asi el
# modelo se entrena con exactamente el tipo de imagen que va a recibir despues desde la camara.
#
# Lee de dos lugares:
#   - Entrena/   : el dataset original de fotos (99 por vocal, una sola sesion)
#   - Capturas/  : lo que se haya grabado con lector.py (puede no existir todavia)
#
# Se importa recortar_mano de recorte.py (y no se copia el codigo aca) justamente para que no
# puedan quedar desalineados: si cambia el recorte, cambia en los dos lados a la vez.
#
# Uso:
#   python preparar_recortes.py
# -----------------------------------------------------------------------------------------

import os

import cv2
import mediapipe as mp

from recorte import recortar_mano

ORIGENES = ["Entrena", "Capturas"]
DESTINO = "RecortesCrudos"
CLASES = ["A", "E", "I", "O", "U"]

# static_image_mode=True porque son fotos sueltas, no un video: MediaPipe no intenta
# "seguir" la mano de un frame al siguiente, hace una deteccion completa en cada imagen.
manos_mp = mp.solutions.hands
detector = manos_mp.Hands(static_image_mode=True, min_detection_confidence=0.5)

sin_deteccion = 0
generadas = 0

for clase in CLASES:
    os.makedirs(os.path.join(DESTINO, clase), exist_ok=True)
    generadas_clase = 0

    for origen in ORIGENES:
        carpeta = os.path.join(origen, clase)
        # Capturas/ puede no existir todavia (si nunca se corrio lector.py).
        if not os.path.isdir(carpeta):
            continue

        for nombre in sorted(os.listdir(carpeta)):
            img = cv2.imread(os.path.join(carpeta, nombre))
            if img is None:
                continue

            resultado = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not resultado.multi_hand_landmarks:
                sin_deteccion += 1
                continue

            landmarks = resultado.multi_hand_landmarks[0].landmark
            recorte, _ = recortar_mano(img, landmarks)
            if recorte is None:
                sin_deteccion += 1
                continue

            cv2.imwrite(os.path.join(DESTINO, clase, nombre), recorte)
            generadas += 1
            generadas_clase += 1

    print(f"{clase}: {generadas_clase} recortes")

print(f"\nRecortes generados: {generadas}, fotos sin deteccion de mano: {sin_deteccion}")
if generadas == 0:
    print("ATENCION: no se genero ningun recorte. Verificar que exista Entrena/ o Capturas/.")
