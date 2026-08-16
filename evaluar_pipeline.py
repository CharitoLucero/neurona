# -----------------------------------------------------------------------------------------
# EVALUAR EL MODELO SOBRE EL SET DE VALIDACION
# -----------------------------------------------------------------------------------------
# Mide cuantas fotos de ValidacionRecorte/ acierta el modelo, en total y por vocal, y lista
# los errores (que letra predijo en lugar de la correcta). Sirve para saber si un
# reentrenamiento mejoro o empeoro, y para ver que vocales se confunden entre si.
#
# Nota: aca NO se vuelve a pasar la imagen por MediaPipe, porque las fotos de
# ValidacionRecorte/ YA son recortes de mano de 200x200 hechos por preparar_recortes.py.
# Volver a detectar y recortar sobre un recorte daria un encuadre distinto al del
# entrenamiento (ese error fue justo el que hacia fallar al proyecto; ver DOCUMENTACION.md
# seccion 12). Lo unico que falta hacerles es pasarlas a RGB y reescalar a [0,1], igual que
# hace ImageDataGenerator al entrenar.
#
# Uso:
#   python evaluar_pipeline.py
# -----------------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from inferencia import CLASES, RUTA_MODELO, RUTA_PESOS

# Se cargan las mismas rutas que usa inferencia.py, para evaluar exactamente el modelo que
# esta en uso y no una version vieja por accidente.
modelo = load_model(RUTA_MODELO, compile=False)
modelo.load_weights(RUTA_PESOS)

CARPETA_VALIDACION = "ValidacionRecorte"

correctas = 0
total = 0
por_clase = {c: [0, 0] for c in CLASES}  # {letra: [aciertos, total]}
confusiones = {}  # {(letra_real, letra_predicha): cantidad}

for letra in CLASES:
    carpeta = os.path.join(CARPETA_VALIDACION, letra)
    for nombre in sorted(os.listdir(carpeta)):
        img = cv2.imread(os.path.join(carpeta, nombre))
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = np.expand_dims(img_to_array(img_rgb) / 255.0, axis=0)

        probabilidades = modelo.predict(x, verbose=0)[0]
        prediccion = CLASES[int(np.argmax(probabilidades))]

        total += 1
        por_clase[letra][1] += 1
        if prediccion == letra:
            correctas += 1
            por_clase[letra][0] += 1
        else:
            confusiones[(letra, prediccion)] = confusiones.get((letra, prediccion), 0) + 1
            print(f"{letra} {nombre} -> {prediccion} ({probabilidades.max():.2f}) MAL")

print(f"\nAciertos: {correctas}/{total} = {correctas/total:.0%}")

print("\nPor vocal:")
for c in CLASES:
    aciertos, cantidad = por_clase[c]
    if cantidad:
        print(f"  {c}: {aciertos}/{cantidad} = {aciertos/cantidad:.0%}")

if confusiones:
    print("\nConfusiones mas frecuentes (real -> predicha):")
    for (real, predicha), cantidad in sorted(confusiones.items(), key=lambda x: -x[1]):
        print(f"  {real} -> {predicha}: {cantidad}")
