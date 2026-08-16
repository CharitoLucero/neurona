# -----------------------------------------------------------------------------------------
# RECORTE DE LA MANO (unica fuente de verdad)
# -----------------------------------------------------------------------------------------
# Este modulo define COMO se recorta la mano de una imagen. Lo importan tanto
# preparar_recortes.py (para armar el dataset de entrenamiento) como inferencia.py (para
# predecir en vivo), justamente para que el encuadre de entrenamiento y el de prediccion
# sean identicos. Cuando no lo eran, el modelo fallaba casi siempre en vivo aunque el
# accuracy de validacion diera bien (ver DOCUMENTACION.md, secciones 12 y 15).
#
# Idea clave: el recorte se calcula a partir del tamaño de la mano detectada (los 21
# landmarks de MediaPipe), NO como una ventana fija de 200x200 pixeles. Una ventana fija
# depende de la resolucion de la imagen y de la distancia a la camara: en las fotos
# originales (de hasta 5120x3840) 200x200 px es apenas un parche de piel, mientras que en un
# frame de webcam (640x480) abarca casi toda la mano. Escalando el recorte al tamaño de la
# mano, el resultado se ve igual en cualquier resolucion y a cualquier distancia.
# -----------------------------------------------------------------------------------------

import cv2

# Cuanto espacio extra se deja alrededor de la mano, como fraccion del lado de la mano.
# 0.25 = 25% de margen por lado (la mano no queda pegada al borde del recorte).
MARGEN = 0.25

# Lado (en pixeles) del recorte final que recibe la red. Debe coincidir con altura/longitud
# en los scripts de entrenamiento.
LADO_SALIDA = 200


def recortar_mano(frame_bgr, landmarks, lado_salida=LADO_SALIDA):
    """
    Recorta la mano de un frame y la devuelve como imagen cuadrada de lado_salida x lado_salida.

    Parametros:
      - frame_bgr: imagen de OpenCV en BGR (cualquier resolucion).
      - landmarks: los 21 puntos de la mano que devuelve MediaPipe (coordenadas relativas 0..1).

    Devuelve (recorte, caja):
      - recorte: imagen BGR de lado_salida x lado_salida, o None si la caja quedo vacia.
      - caja: (x1, y1, x2, y2) en pixeles del frame original, para poder dibujar el recuadro.
        Puede incluir coordenadas fuera del frame si la mano esta cortada por el borde.
    """
    alto_frame, ancho_frame = frame_bgr.shape[:2]

    # Caja que envuelve todos los puntos de la mano, en pixeles.
    xs = [p.x * ancho_frame for p in landmarks]
    ys = [p.y * alto_frame for p in landmarks]

    centro_x = (min(xs) + max(xs)) / 2
    centro_y = (min(ys) + max(ys)) / 2

    # Se usa un recuadro CUADRADO (el lado mas largo de la mano + margen) para que al
    # redimensionar a lado_salida x lado_salida la seña no se deforme.
    lado = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * MARGEN)

    x1 = int(round(centro_x - lado / 2))
    y1 = int(round(centro_y - lado / 2))
    x2 = int(round(centro_x + lado / 2))
    y2 = int(round(centro_y + lado / 2))

    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)

    # Si la mano esta cerca del borde, el cuadrado se sale del frame. En vez de recortarlo
    # (lo que deformaria la seña al redimensionar), se rellena con negro la parte que falta,
    # asi el recorte sigue siendo cuadrado y la mano mantiene su proporcion.
    relleno_izq = max(0, -x1)
    relleno_arriba = max(0, -y1)
    relleno_der = max(0, x2 - ancho_frame)
    relleno_abajo = max(0, y2 - alto_frame)

    recorte = frame_bgr[max(0, y1):min(alto_frame, y2), max(0, x1):min(ancho_frame, x2)]
    if recorte.size == 0:
        return None, (x1, y1, x2, y2)

    if relleno_izq or relleno_arriba or relleno_der or relleno_abajo:
        recorte = cv2.copyMakeBorder(
            recorte,
            relleno_arriba, relleno_abajo, relleno_izq, relleno_der,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    # INTER_AREA da mejor calidad al reducir (fotos grandes); INTER_CUBIC al ampliar.
    interpolacion = cv2.INTER_AREA if recorte.shape[0] > lado_salida else cv2.INTER_CUBIC
    recorte = cv2.resize(recorte, (lado_salida, lado_salida), interpolation=interpolacion)

    return recorte, (x1, y1, x2, y2)
