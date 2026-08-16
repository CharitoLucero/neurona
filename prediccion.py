# -----------------------------------------------------------------------------------------
# PREDICCION EN VIVO (aplicacion de escritorio)
# -----------------------------------------------------------------------------------------
# Muestra el video de la camara y, sobre cada frame, dibuja el recuadro de la mano
# detectada junto con la vocal reconocida (o "LETRA DESCONOCIDA" si la confianza es baja).
#
# Toda la logica de deteccion de mano + prediccion vive en inferencia.py, para poder
# reutilizarla tambien desde la API (api.py) sin duplicar codigo.
# -----------------------------------------------------------------------------------------

import cv2

from inferencia import predecir_desde_frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        # La camara no entrego un frame valido (por ejemplo, si se desconecto): salimos.
        break

    resultado = predecir_desde_frame(frame)

    # Depuracion temporal: muestra en la consola la probabilidad de cada vocal para
    # entender por que el modelo elige (o descarta) una letra en particular.
    if resultado["probabilidades"] is not None:
        print(resultado["probabilidades"])

    if resultado["caja"] is not None:
        x1, y1, x2, y2 = resultado["caja"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        if resultado["letra"] is not None:
            texto = "{} ({:.0f}%)".format(resultado["letra"], resultado["confianza"] * 100)
        else:
            texto = "LETRA DESCONOCIDA"

        cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()


