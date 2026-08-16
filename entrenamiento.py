# #------------------------------ Entrenamiento de la red (CNN) ---------------------------------------
# Entrena con los recortes de EntrenamientoRecorte/, armados por preparar_recortes.py +
# repartir_recortes.py. Esos recortes se generan con la MISMA funcion
# (recorte.recortar_mano) que usa inferencia.py al predecir, asi el modelo aprende
# exactamente con el tipo de imagen que va a recibir en vivo desde la camara.
#
# Por que importa: las dos causas por las que el proyecto fallaba en vivo eran (1) el dataset
# original tenia el fondo borrado (fondo negro) y (2) el recorte de prediccion no coincidia
# con el encuadre de entrenamiento. Ver DOCUMENTACION.md, secciones 12 y 15.
#
# Uso (recomendado con UTF-8 para que la barra de progreso de Keras no rompa el log en Windows):
#   set PYTHONUTF8=1
#   python entrenamiento.py
import tensorflow as tf
import tensorflow.keras.optimizers


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

K.clear_session()

datos_entrenamiento = "E:/red neuronal/EntrenamientoRecorte"
datos_validacion = "E:/red neuronal/ValidacionRecorte"

iteraciones = 20
altura, longitud = 200, 200
batch_size = 16
filtrosconv1 = 32
filtrosconv2 = 64
filtrosconv3 = 128

tam_filtro1 = (4, 4)
tam_filtro2 = (3, 3)
tam_filtro3 = (2, 2)
tam_pool = (2, 2)
clases = 5
lr = 0.0005

preprocesamiento_entre = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

preprocesamiento_vali = ImageDataGenerator(
    rescale=1./255
)

generador_entrena = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical',
)

generador_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

print("Mapeo de clases (debe ser A=0, E=1, I=2, O=3, U=4):", generador_entrena.class_indices)

pasos = generador_entrena.samples // batch_size
pasos_validacion = generador_validacion.samples // batch_size

# flow_from_directory devuelve un iterador clasico de Keras que en esta version de
# TensorFlow/Keras no se reinicia solo al terminar una epoca (el fit de Keras 3 lo agota
# y explota en la 2da epoca). Se envuelve en un tf.data.Dataset con .repeat() para que
# vuelva a arrancar desde el principio en cada epoca.
firma_salida = (
    tf.TensorSpec(shape=(None, altura, longitud, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None, clases), dtype=tf.float32),
)
imagen_entrena = tf.data.Dataset.from_generator(lambda: generador_entrena, output_signature=firma_salida).repeat()
imagen_validacion = tf.data.Dataset.from_generator(lambda: generador_validacion, output_signature=firma_salida).repeat()

cnn = Sequential()

cnn.add(Conv2D(filtrosconv1, tam_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Conv2D(filtrosconv2, tam_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Conv2D(filtrosconv3, tam_filtro3, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

cnn.add(Flatten())
cnn.add(Dense(640, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

optimizar = tensorflow.keras.optimizers.Adam(learning_rate=lr)
cnn.compile(loss='categorical_crossentropy', optimizer=optimizar, metrics=['accuracy'])

cnn.fit(imagen_entrena, steps_per_epoch=pasos, epochs=iteraciones, validation_data=imagen_validacion, validation_steps=pasos_validacion)

cnn.save('modeloVocales_mano.keras')
cnn.save_weights('pesosVocales_mano.weights.h5')
