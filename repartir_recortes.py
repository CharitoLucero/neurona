# -----------------------------------------------------------------------------------------
# REPARTIR LOS RECORTES EN ENTRENAMIENTO Y VALIDACION
# -----------------------------------------------------------------------------------------
# Toma RecortesCrudos/ (generado por preparar_recortes.py) y lo divide en:
#   - EntrenamientoRecorte/  (80% de las fotos de cada vocal)
#   - ValidacionRecorte/     (20% restante)
#
# Hace falta porque el proyecto no traia una division real: la carpeta Valida/ que venia en el
# repo era una copia identica de Entrena/ (mismo hash archivo por archivo), asi que validar
# contra ella era validar con las mismas fotos del entrenamiento. Por eso se borro y el
# reparto se hace aca, a partir de una sola carpeta fuente.
#
# IMPORTANTE - por que el reparto es POR BLOQUES y no al azar:
# Las fotos del dataset parecen tomadas en rafaga, asi que dos fotos con numeros
# consecutivos son casi identicas (se midio: la diferencia media entre fotos consecutivas es
# menos de la mitad que entre fotos alejadas). Con un reparto al azar, la foto A5 podia
# quedar en entrenamiento y A6 -su casi gemela- en validacion. Aunque son archivos distintos
# (el modelo nunca ve la MISMA foto en los dos lados), al modelo no le hace falta generalizar
# para acertarlas: le alcanza con reconocer esa toma puntual. Eso infla el accuracy de
# validacion y explica que diera ~90% mientras en vivo fallaba casi siempre.
#
# Repartiendo por bloques (las ultimas N fotos de cada vocal, en orden numerico, van a
# validacion) las fotos de validacion no son gemelas de ninguna de entrenamiento, y el
# numero que sale es mucho mas honesto.
#
# Aun asi, la prueba definitiva es capturar fotos NUEVAS con la webcam en otro momento/luz:
# todo este dataset es de una sola persona, un solo fondo y una sola sesion.
#
# Uso:
#   python repartir_recortes.py
# -----------------------------------------------------------------------------------------

import os
import re
import shutil

ORIGEN = "RecortesCrudos"
DESTINO_TRAIN = "EntrenamientoRecorte"
DESTINO_VAL = "ValidacionRecorte"
CLASES = ["A", "E", "I", "O", "U"]
PROPORCION_VAL = 0.2

def numero_de(nombre):
    """
    Extrae el numero del nombre del archivo (A7.jpg -> 7) para poder ordenarlos como los
    tomo la camara. Ordenar como texto pondria A10 antes que A2, y entonces el bloque de
    validacion no serian fotos consecutivas.
    """
    digitos = re.findall(r"\d+", nombre)
    return int(digitos[0]) if digitos else 0


for clase in CLASES:
    # Orden numerico = orden en que se tomaron las fotos.
    archivos = sorted(os.listdir(os.path.join(ORIGEN, clase)), key=numero_de)

    cantidad_val = max(1, round(len(archivos) * PROPORCION_VAL))
    # El bloque final (fotos con los numeros mas altos) va entero a validacion.
    entrenamiento = archivos[:-cantidad_val]
    validacion = archivos[-cantidad_val:]

    os.makedirs(os.path.join(DESTINO_TRAIN, clase), exist_ok=True)
    os.makedirs(os.path.join(DESTINO_VAL, clase), exist_ok=True)

    for nombre in entrenamiento:
        shutil.copy(os.path.join(ORIGEN, clase, nombre), os.path.join(DESTINO_TRAIN, clase, nombre))
    for nombre in validacion:
        shutil.copy(os.path.join(ORIGEN, clase, nombre), os.path.join(DESTINO_VAL, clase, nombre))

    print(f"{clase}: {len(entrenamiento)} entrenamiento / {len(validacion)} validacion "
          f"(de {len(archivos)} en total) | validacion = fotos {numero_de(validacion[0])} a {numero_de(validacion[-1])}")
