#PREDICCIONES
import cv2
import mediapipe as mp
import os
import numpy as np
import tensorflow as tf


from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

#------------------Cargamos el modelo-----------------------------------------

modelo = "E:/red neuronal/modeloVocales.keras"
peso = 'E:/red neuronal/pesosVocales.weights.h5'
cnn = load_model(modelo)
cnn.load_weights(peso)


direccion = "E:/red neuronal/Validacion"
dire_img = os.listdir(direccion)
print('Nombres ',dire_img)


# Leemos la camara
cap = cv2.VideoCapture(0)


#Creamos objeto que va almacenar la deteccion  y el seguimiento de las manos

clase_manos = mp.solutions.hands
manos = clase_manos.Hands()     # Primer parametro, FALSE para que no haga la deteccion 24/7
                                # solo har deteccion cuando hay una cofianza alta
                                 #Segundo parametro: numero maximo de manos
                                # Tercer parametro: confianza minima para deteccion
                                # Cuarto parametro: confianza minima para seguimiento'''

#Metodo para dibujar las manos
dibujo = mp.solutions.drawing_utils #Con este metodo dibujamos 21 puntos de las manos

while (1):
  ret, frame = cap.read()

  color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  copia = frame.copy()
  resultado = manos.process(color)
  posiciones = [] # En esta lista vamos a almacena las coordenadas de ls puntos
  #print(resultado.multi_hand_landmarks) # Si queremos ver si existe la deteccion

  if resultado.multi_hand_landmarks: # Si hay  en los resultados entramos al if
   for mano in resultado.multi_hand_landmarks: #Buscamos la mano dentro de la lista de manos que nos da el descriptor
      for id, lm in enumerate(mano.landmark):  #Vamos a obtener la informacion de cada mano encontrada por el ID
        #print(id, lm) #Como nos entregan decimales (Proporcion de la imagen) debemos pasarlo a pixeles
        altura, ancho, c = frame.shape #Extraemos el ancho y el alto de los fotogramas para multiplicarlos por la proporcion
        corx, cory = int(lm.x*ancho), int(lm.y*altura) #Extraemos la ubicacon de cada punto que pertenece a la mano en coordinadas
        posiciones.append([id, corx, cory])
        dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
      if len(posiciones) != 0:
        pto_i1 = posiciones[4] #5 dedos: 4 | 0 dedos : 3 | 1 dedo: 2 | 2 dedos: 3 | 3 dedos: 4 | 4 dedos: 8
        pto_i2 = posiciones[20] #5 dedos: 20 | 0 dedos : 17 | 1 dedo: 17 | 2 dedos: 20 | 3 dedos: 20 | 4 dedos: 20
        pto_i3 = posiciones[12] #5 dedos: 12 | 0 dedos : 10 | 1 dedo: 20 | 2 dedos: 16 | 3 dedos: 12 | 4 dedos: 12
        pto_i4 = posiciones[0] #5 dedos: 0 | 0 dedos : 0 | 1 dedo: 0 | 2 dedos: 0 | 3 dedos: 0 | 4 dedos: 0
        pto_i5 = posiciones[9] #Punto central

        # x1, y1 = (pto_i5[1]-100), (pto_i5[2]-100) #Obtenemos el punto inicial y las longitudes
        # ancho, alto  = (x1+200), (y1+200)
        # x2, y2 = x1 + ancho, y1 + alto
        # dedos_reg = copia[y1:y2, x1:x2]
        # #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation = cv2.INTER_CUBIC) #Redimencionamos las fotos

        x1 = max(0, pto_i5[1] - 100)
        y1 = max(0, pto_i5[2] - 100)
        x2 = min(frame.shape[1], x1 + 200)
        y2 = min(frame.shape[0], y1 + 200)
        dedos_reg = copia[y1:y2, x1:x2]

  
        x = img_to_array(dedos_reg)
        x = np.expand_dims(x, axis=0)
        vector = cnn.predict(x)
        resultado = vector[0] #[1,0] [0,1]
        respuesta = np.argmax(resultado)

        if respuesta == 0:
          print('resultado')
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #   cv2.putText(frame, 'A', format(dire_img [0]),(x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
          cv2.putText(frame, format(dire_img[0]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
        elif respuesta == 1:
          print('resultado')
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #   cv2.putText(frame, 'E', format(dire_img [1]),(x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
          cv2.putText(frame, format(dire_img[1]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
        elif respuesta == 2:
          print('resultado')
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #   cv2.putText(frame, 'I', format(dire_img [2]),(x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
          cv2.putText(frame, format(dire_img[2]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
        elif respuesta == 3:
          print('resultado')
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #   cv2.putText(frame, 'O', format(dire_img [3]),(x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
          cv2.putText(frame, format(dire_img[3]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
        elif respuesta == 4:
          print('resultado')
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
        #   cv2.putText(frame, 'U', format(dire_img [4]),(x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
          cv2.putText(frame, format(dire_img[4]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

        else:
          # cv2.putText, (frame,'LETRA DESCONOCIDA', (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
          cv2.putText(frame, 'LETRA DESCONOCIDA', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 1, cv2.LINE_AA)








  cv2.imshow("Video", frame)
  k = cv2.waitKey(1)
  if k == 27:
    break
cap.release()
cv2.destroyAllWindows()

