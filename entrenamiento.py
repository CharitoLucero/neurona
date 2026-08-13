# #------------------------------ Crear modelo y entrenarlo ---------------------------------------
import tensorflow.keras.optimizers


from tensorflow.keras.preprocessing.image import ImageDataGenerator #Nos ayuda a preprocesar las imagenes que le entreguemos al modelo
from tensorflow.keras.models import Sequential #Nos permite hacer redes neuronales secuenciales
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, Conv2D, MaxPooling2D  #Capas para hacer las convoluciones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K  #Si hay una sesion de keras, lo cerramos para tener todo limpio

K.clear_session()  #Limpiamos todo

#---------------------------------Importamos las fotos tomadas-----------------------------
datos_entrenamiento = "E:/red neuronal/Entrenamiento" #cambiar a ruta propia
datos_validacion = "E:/red neuronal/Validacion" #cambiar a ruta propia

#PARAMETROS

iteraciones = 20 #Numero de iteraciones para ajustar nustro modelo
altura, longitud = 200, 200 #Tamaño de las imagenes de entrenamiento
batch_size = 1 #Numero de veces que se va a procesar la informacion en cada iteracion
filtrosconv1 = 32 #Numero de filtros en la primer capa de convolucion
filtrosconv2 = 64 #Numero de filtros en la segunda capa de convolucion
filtrosconv3 = 128 #Numero de filtros en la tercer capa de convolucion
#Esta capa se agrega para extraer más informacion



tam_filtro1 = (4,4) #Tamaño del filtro en la primer capa de convolucion
tam_filtro2 = (3,3) #Tamaño del filtro en la segunda capa de convolucion
tam_filtro3 = (2,2) #Tamaño del filtro en la tercer capa de convolucion
tam_pool = (2,2) #Tamaño del filtro en max popling
clases = 5 #Numero de clases 5 vocales (5 dedos y o dedos)
lr = 0.0005 #Ajuste de la red nuronal para acercarse a una solucion optima (Tasa de aprendizaje)


#Pre-procesamiento de las imagenes

preprocesamiento_entre = ImageDataGenerator(
    rescale=1./255,  #Pasar los pixeles de 0 a 255 | 0 a 1
    shear_range=0.3, #Observar nuestras imagenes inclinadas para un mejor entrenamiento
    zoom_range=0.3,  #Genera imagenes cnn 20px para un mejor entrenamiento
    horizontal_flip=True #Invierte las imagenes para mejorar entrenamiento
)

preprocesamiento_vali = ImageDataGenerator(
    rescale=1./255
)

imagen_entrena = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,      #Va a tocar las fotos que va a almacenar
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical', #Clasificacion categorica por clases

)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)


# Calcular pasos por época dinámicamente según la cantidad real de imágenes
pasos = imagen_entrena.samples // batch_size
pasos_validacion = imagen_validacion.samples // batch_size



#Creamos la red neuronal convolucional (CNN)
cnn = Sequential() #Red neuronal secuencial

#Agregamos filtros con el fin de volver nuestra imagen muy profunda pero pequeña
cnn.add(Conv2D(filtrosconv1, tam_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu')) #Es una convolucion y realizamos config
cnn.add(MaxPooling2D(pool_size=tam_pool))   #Despues de la primera capa vamos a tener una capa de max popling y signamos el tamaño
                                            #Max popling es la extracion de caracteristica

cnn.add(Conv2D(filtrosconv2, tam_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

#Nueva Capa
cnn.add(Conv2D(filtrosconv3, tam_filtro3, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))


#Convertir esa imagen profunda en una plana, para tener 1 dimension con toda la info

cnn.add(Flatten()) #Aplanamos la imagen
cnn.add(Dense(640, activation='relu')) #Agregamos 426 neuronas
cnn.add(Dropout(0.5))  #Apagamos el 50% de las neuronas en la funcion anterior para no sobre
cnn.add(Dense(clases, activation='softmax')) #Ultima capa, la que dice la probabilidad de queuna imagen sea alguna de las clases

#Agregamos parametros para optimizar el modelo
#Durante el entrenamiento tenga una autoevaluacion que se optimice con Adam, y la metrica sera accuracy

optimizar = tensorflow.keras.optimizers.Adam(learning_rate= lr)
cnn.compile(loss='categorical_crossentropy', optimizer=optimizar, metrics=['accuracy'])

#Entrenamos nuestra red
cnn.fit(imagen_entrena, steps_per_epoch=pasos, epochs=iteraciones, validation_data=imagen_validacion, validation_steps=pasos_validacion)

# Guardamos el modelo
cnn.save('modeloVocales.keras')
cnn.save_weights('pesosVocales.weights.h5')



