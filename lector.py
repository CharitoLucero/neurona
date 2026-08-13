#-----------------------------DECTECTOR DE MANOS--------------------------------------------

'''
Instalar Librerías previamente
Abriir nueva terminal y copiar los siguientes codigos:


# pip install tensorflow 
# pip install keras
# pip install mediapipe
# pip install cv2
# pip install sklearn
'''


#-------------------------------------------------------------------------------------------
#IMPORTACION DE LIBRERIAS

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model


import cv2 #Lector de camara
import mediapipe as mp #lector de manos
import os #Permite moverse entre carpetas


#Crear carpeta donde almacenar entrenamiento

nombre = 'A'
direccion = "E:/red neuronal/Entrenamiento"  #cambiar a ruta propia
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)

#Asignmaos un contador para el nombre de las fotos
cont = 0

#Leemos la camara
cap = cv2.VideoCapture(0)


#Creamo objeto que va almacenar el entrenamiento

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

        x1, y1 = (pto_i5[1]-100), (pto_i5[2]-100) #Obtenemos el punto inicil y las longitudes
        ancho, alto  = (x1+200), (y1+200)
        x2, y2 = x1 + ancho, y1 + alto
        dedos_reg = copia[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
    #   dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolacion = cv2.INTER_CUBIC) #Redimencionamos las fotos
    #   cv2.inwrite(carpeta + '/dedos_{}.jpg'.format(cont), dedos_reg)
    #   cont = cont + 1


  cv2.imshow("Video", frame)
  k = cv2.waitKey(1)
  if k == 27 or cont >= 100:
    break
cap.release()
cv2.destroyAllWindows()
