# -----------------------------------------------------------------------------------------
# CAPTURA DE FOTOS PARA EL DATASET
# -----------------------------------------------------------------------------------------
# Graba fotos de una seña con la webcam, para poder entrenar el modelo con datos nuevos.
#
# Guarda el FRAME COMPLETO de la camara, no el recorte de la mano. Es a proposito: el recorte
# lo hace despues preparar_recortes.py. Si el dia de mañana cambia la forma de recortar (ya
# cambio dos veces, ver DOCUMENTACION.md secciones 12 y 15), alcanza con volver a correr
# preparar_recortes.py sobre estas fotos, sin tener que capturar todo de nuevo.
#
# Uso:
#   python lector.py A                      -> 100 fotos de la vocal A en Capturas/A/
#   python lector.py E --cantidad 150       -> 150 fotos de la vocal E
#   python lector.py I --intervalo 0.6      -> mas lento (mas tiempo para mover la mano)
#
# Controles mientras corre:
#   ESPACIO -> empezar / pausar la captura
#   ESC     -> salir
#
# Recomendacion importante para que el modelo generalice: mientras captura, ir MOVIENDO la
# mano (acercarla, alejarla, girarla, cambiarla de lugar en el cuadro) y hacer varias tomas
# en momentos y luces distintos. El dataset original fallaba justamente por ser una sola
# sesion, con una sola luz y un solo fondo: el modelo memorizaba esa sesion en vez de
# aprender la forma de la seña.
# -----------------------------------------------------------------------------------------

import argparse
import os
import re
import time

import cv2
import mediapipe as mp

from recorte import recortar_mano

CLASES_VALIDAS = ["A", "E", "I", "O", "U"]

# Las fotos capturadas se numeran desde aca en adelante. Arranca en 1000 (y no en 1) para que
# queden siempre por encima de las del dataset original de Entrena/, que van de 1 a 99. Como
# repartir_recortes.py arma el set de validacion tomando el bloque de numeros mas altos, esto
# hace que lo que se valide sea la sesion de captura mas reciente, que es lo mas parecido a
# una prueba honesta que se puede lograr con un dataset chico.
NUMERO_INICIAL = 1000


def parsear_argumentos():
    """Lee los parametros de la terminal, asi no hay que editar el codigo para cada vocal."""
    analizador = argparse.ArgumentParser(
        description="Captura fotos de una seña con la webcam para armar el dataset."
    )
    analizador.add_argument(
        "letra",
        help="Vocal que se esta capturando (A, E, I, O o U)",
    )
    analizador.add_argument(
        "-c", "--cantidad",
        type=int, default=100,
        help="Cuantas fotos capturar (por defecto 100)",
    )
    analizador.add_argument(
        "-d", "--destino",
        default="Capturas",
        help="Carpeta donde guardar las fotos (por defecto Capturas)",
    )
    analizador.add_argument(
        "-i", "--intervalo",
        type=float, default=0.4,
        help="Segundos de espera entre foto y foto (por defecto 0.4). Sirve para tener "
             "tiempo de mover la mano: si se guardan muchas fotos por segundo, salen todas "
             "casi identicas y no le aportan nada al entrenamiento.",
    )
    return analizador.parse_args()


def proximo_numero(carpeta):
    """
    Devuelve el numero con el que hay que seguir numerando las fotos de la carpeta, para no
    sobreescribir las que ya estan (asi se pueden hacer varias sesiones de captura).

    Ademas, que las fotos nuevas queden con numeros mas altos hace que repartir_recortes.py
    -que separa validacion tomando el bloque final- deje afuera la sesion mas reciente. Eso
    es lo mejor que se puede hacer con un dataset chico: validar contra una sesion que el
    modelo no vio nunca.
    """
    if not os.path.isdir(carpeta):
        return NUMERO_INICIAL

    numeros = [NUMERO_INICIAL - 1]
    for nombre in os.listdir(carpeta):
        encontrados = re.findall(r"\d+", nombre)
        if encontrados:
            numeros.append(int(encontrados[-1]))
    return max(numeros) + 1


def dibujar_estado(frame, letra, guardadas, cantidad, capturando, caja):
    """Escribe sobre el video la informacion de lo que esta pasando."""
    if caja is not None:
        # Se dibuja el mismo recuadro que despues va a recortar preparar_recortes.py, asi se
        # ve en pantalla que parte de la imagen es la que realmente va a usar el modelo.
        x1, y1, x2, y2 = caja
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if capturando:
        estado, color = "CAPTURANDO", (0, 200, 0)
    else:
        estado, color = "EN PAUSA - ESPACIO para empezar", (0, 165, 255)

    cv2.putText(frame, f"{letra}  {guardadas}/{cantidad}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, estado, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Movete: acerca, aleja y gira la mano", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def main():
    argumentos = parsear_argumentos()
    letra = argumentos.letra.upper()

    if letra not in CLASES_VALIDAS:
        print(f"ERROR: '{letra}' no es una de las clases del modelo ({', '.join(CLASES_VALIDAS)}).")
        return

    carpeta = os.path.join(argumentos.destino, letra)
    os.makedirs(carpeta, exist_ok=True)

    numero = proximo_numero(carpeta)
    print(f"Guardando en: {carpeta}")
    print(f"Empezando a numerar desde {numero} (para no sobreescribir lo que ya habia)")
    print("ESPACIO = empezar/pausar | ESC = salir")

    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        print("ERROR: no se pudo abrir la camara.")
        return

    clase_manos = mp.solutions.hands
    detector = clase_manos.Hands(max_num_hands=1, min_detection_confidence=0.7)
    dibujo = mp.solutions.drawing_utils

    guardadas = 0
    capturando = False
    momento_ultima_foto = 0.0

    while guardadas < argumentos.cantidad:
        ok, frame = camara.read()
        if not ok:
            # La camara no entrego un frame (se desconecto, o la esta usando otro programa).
            print("ERROR: no se pudo leer un frame de la camara.")
            break

        # Se guarda una copia limpia ANTES de dibujarle encima, para que las fotos del
        # dataset no tengan los puntos ni el recuadro pintados.
        frame_limpio = frame.copy()

        resultado = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        caja = None
        if resultado.multi_hand_landmarks:
            mano = resultado.multi_hand_landmarks[0]
            dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            # Se calcula el recorte con la MISMA funcion que usa la prediccion, solo para
            # mostrar el recuadro en pantalla y para confirmar que la mano se ve completa.
            recorte, caja = recortar_mano(frame_limpio, mano.landmark)

            hay_tiempo_cumplido = (time.time() - momento_ultima_foto) >= argumentos.intervalo
            if capturando and recorte is not None and hay_tiempo_cumplido:
                archivo = os.path.join(carpeta, f"{letra}_{numero}.jpg")
                cv2.imwrite(archivo, frame_limpio)
                numero += 1
                guardadas += 1
                momento_ultima_foto = time.time()

        dibujar_estado(frame, letra, guardadas, argumentos.cantidad, capturando, caja)
        cv2.imshow("Captura de dataset", frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 27:  # ESC
            break
        if tecla == 32:  # ESPACIO
            capturando = not capturando

    camara.release()
    cv2.destroyAllWindows()

    print(f"\nListo: {guardadas} fotos nuevas de '{letra}' en {carpeta}")
    if guardadas:
        print("Para usarlas: python preparar_recortes.py && python repartir_recortes.py && python entrenamiento.py")


if __name__ == "__main__":
    main()
