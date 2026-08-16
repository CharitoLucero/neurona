# Documentación del proyecto — Lector de lengua de señas (vocales)

> Este documento describe, paso a paso, cómo está construido el proyecto tal como existe hoy: qué hace cada archivo, cómo se relacionan entre sí, cómo ejecutar cada etapa, y qué limitaciones y bugs conocidos tiene. Sirve como base antes de refactorizarlo y exponerlo como API.

## 1. Qué hace el proyecto

Es un lector de señas con la mano que reconoce **5 vocales** (A, E, I, O, U) a partir de la posición de los dedos, usando la cámara web. No reconoce el alfabeto completo de lengua de señas, solo estas 5 clases.

El flujo completo tiene 3 etapas, una por script:

1. **Captura de datos** (`lector.py`) → genera las fotos de entrenamiento/validación.
2. **Entrenamiento** (`entrenamiento.py`) → entrena una red neuronal convolucional (CNN) con esas fotos y guarda el modelo.
3. **Predicción en vivo** (`prediccion.py`) → carga el modelo entrenado y predice la vocal en tiempo real desde la cámara.

## 2. Estructura de carpetas y archivos

> Esta es la estructura **actual**, después de la limpieza descrita en la sección 18. Las secciones 3 a 5 describen el proyecto tal como estaba originalmente (con archivos que hoy ya no existen); se conservan porque explican de dónde viene cada cosa.

```
red neuronal/
│
│   # --- Núcleo: lo que se usa para predecir ---
├── recorte.py                      # Cómo se recorta la mano. UNICA fuente de verdad del encuadre.
├── inferencia.py                   # Carga el modelo y predice una vocal a partir de un frame
├── prediccion.py                   # App de escritorio: cámara en vivo con el resultado dibujado
├── api.py                          # API REST (FastAPI) que expone la misma predicción por HTTP
│
│   # --- Preparar datos y entrenar ---
├── lector.py                       # Captura fotos con la cámara para armar el dataset
├── preparar_recortes.py            # Entrena/ -> RecortesCrudos/ (aplica el recorte de recorte.py)
├── repartir_recortes.py            # RecortesCrudos/ -> EntrenamientoRecorte/ + ValidacionRecorte/
├── entrenamiento.py                # Entrena la CNN con esos recortes
├── evaluar_pipeline.py             # Mide aciertos del modelo sobre ValidacionRecorte/
│
│   # --- Modelo en uso ---
├── modeloVocales_mano.keras        # Modelo entrenado (arquitectura + pesos)
├── pesosVocales_mano.weights.h5    # Pesos entrenados
│
│   # --- Datos ---
├── Entrena/                        # FUENTE: 99 fotos crudas por vocal (A/E/I/O/U), fondo real
│
│   # --- Otros ---
├── DOCUMENTACION.md                # Este archivo
├── requirements.txt                # Dependencias fijadas
└── .gitignore
```

**Carpetas derivadas (no están en el repo, se regeneran con los scripts):** `RecortesCrudos/`, `EntrenamientoRecorte/` y `ValidacionRecorte/` salen de `Entrena/` corriendo `preparar_recortes.py` y después `repartir_recortes.py`. Se borraron en la limpieza porque son reproducibles en un par de minutos.

**El orden importa:** para reentrenar de cero hay que correr, en este orden, `preparar_recortes.py` → `repartir_recortes.py` → `entrenamiento.py` → `evaluar_pipeline.py`.

## 3. Etapa 1 — Captura de datos (`lector.py`)

**Propósito:** generar las imágenes de una mano haciendo una seña, recortadas alrededor de la mano, para usarlas luego como dataset de entrenamiento.

**Paso a paso de lo que hace el script:**

1. Define en qué carpeta va a guardar las fotos: `nombre = 'A'` (la letra que se está capturando) y `direccion = "E:/red neuronal/Entrenamiento"`. Si la subcarpeta `Entrenamiento/A` no existe, la crea.
2. Abre la cámara web (`cv2.VideoCapture(0)`).
3. Crea el detector de manos de MediaPipe (`mp.solutions.hands.Hands()`), que va a devolver 21 puntos (landmarks) de la mano por cada fotograma donde detecte una mano.
4. Entra en un bucle infinito (`while(1)`) que, en cada fotograma:
   - Lee el frame de la cámara y lo convierte de BGR a RGB (MediaPipe espera RGB).
   - Le pasa el frame a MediaPipe (`manos.process(color)`) para detectar manos.
   - Si detecta una mano, recorre sus 21 landmarks, convierte cada uno de coordenadas relativas (0 a 1) a píxeles reales del frame, y dibuja los puntos y conexiones sobre el frame (`dibujo.draw_landmarks`).
   - Toma el punto 9 (centro de la palma) como referencia y calcula un cuadro de 200×200 píxeles alrededor de él (`x1,y1` a `x2,y2`).
   - Dibuja ese cuadro en el frame que se muestra en pantalla (`cv2.rectangle`).
   - **Las líneas que efectivamente guardarían la foto recortada están comentadas** (`cv2.resize`, `cv2.imwrite`, incremento de `cont`). Tal como está el archivo, el script muestra el recuadro pero **no guarda ninguna imagen nueva**.
5. Muestra el video en una ventana (`cv2.imshow`). El bucle termina si se presiona ESC (`k == 27`) o si `cont >= 100` (aunque `cont` nunca se incrementa porque el guardado está deshabilitado, así que en la práctica solo se sale con ESC).

**Cómo se ejecutaba originalmente (para regenerar el dataset):**

1. Editar `nombre` con la letra que se quiere capturar (ej. `'A'`).
2. Descomentar las 3 líneas de guardado (`dedos_reg = cv2.resize(...)`, `cv2.imwrite(...)`, `cont = cont + 1`) y corregir el typo `cv2.inwrite` → `cv2.imwrite`.
3. Correr `python lector.py`, poner la mano haciendo la seña de esa vocal frente a la cámara, dejar que capture ~100-350 fotos moviendo la mano/ángulo/distancia.
4. Presionar ESC para terminar, y repetir para cada una de las 5 vocales.
5. Repetir todo el proceso una segunda vez apuntando a `Validacion` en lugar de `Entrenamiento` (con menos fotos, ej. 150).

**Problemas conocidos de este script** (ver sección 6 para la lista completa):
- Importa librerías de un proyecto de NLP que no usa (`Tokenizer`, `pad_sequences`, `Conv1D`, `Embedding`, `train_test_split`, `pandas`, `matplotlib`).
- No valida que la lectura de cámara (`ret`) haya sido exitosa.
- El recorte de la mano no se limita a los bordes del frame (puede fallar si la mano está cerca del borde).
- Hay que editar el código a mano para cada letra; no es parametrizable desde la terminal.

## 4. Etapa 2 — Entrenamiento (`entrenamiento.py`)

**Propósito:** entrenar una red neuronal convolucional (CNN) que, dada una foto de 200×200 de una mano, prediga cuál de las 5 vocales representa.

**Paso a paso de lo que hace el script:**

1. Limpia cualquier sesión previa de Keras (`K.clear_session()`).
2. Define las rutas del dataset: `Entrenamiento` y `Validacion`.
3. Define los hiperparámetros:
   - `iteraciones = 20` (épocas de entrenamiento).
   - `altura, longitud = 200, 200` (tamaño al que se redimensionan las imágenes).
   - `batch_size = 1` (cuántas imágenes se procesan juntas antes de actualizar los pesos — **muy bajo**, ver sección 6).
   - 3 capas convolucionales con 32, 64 y 128 filtros respectivamente.
   - `clases = 5`, `lr = 0.0005` (tasa de aprendizaje).
4. Define el **preprocesamiento** con `ImageDataGenerator`:
   - Entrenamiento: reescala los píxeles de [0,255] a [0,1], y aplica aumentos de datos aleatorios (inclinación, zoom, espejo horizontal) para que el modelo generalice mejor con pocas fotos.
   - Validación: solo reescala, sin aumentos (para medir el desempeño real, sin trucos).
5. Carga las imágenes desde las carpetas con `flow_from_directory`, que además **asigna automáticamente un índice numérico a cada clase según el orden de las carpetas** (esto es importante: ese orden debe coincidir con el que usa `prediccion.py` al leer resultados, y hoy no hay garantía de que coincida — ver sección 6).
6. Calcula cuántos pasos por época corresponden según la cantidad real de imágenes encontradas (`samples // batch_size`).
7. Construye la CNN con Keras `Sequential`:
   - Conv2D(32 filtros, 4×4) → MaxPooling2D
   - Conv2D(64 filtros, 3×3) → MaxPooling2D
   - Conv2D(128 filtros, 2×2) → MaxPooling2D
   - Flatten (aplana la imagen a un vector)
   - Dense(640, relu) → Dropout(0.5) (apaga la mitad de las neuronas al azar en cada paso, para evitar sobreajuste)
   - Dense(5, softmax) (capa final: probabilidad de cada una de las 5 vocales)
8. Compila el modelo con optimizador Adam y función de pérdida `categorical_crossentropy` (estándar para clasificación multi-clase con una sola clase correcta por imagen).
9. Entrena con `cnn.fit(...)` durante 20 épocas, evaluando contra el set de validación en cada una.
10. Guarda el modelo completo en `modeloVocales.keras` y los pesos por separado en `pesosVocales.weights.h5`.

**Cómo se ejecuta:**

```
python entrenamiento.py
```

Requiere que `Entrenamiento/` y `Validacion/` ya tengan las 5 subcarpetas con imágenes (generadas en la etapa 1).

**Problemas conocidos de este script** (ver sección 6 para la lista completa):
- `batch_size = 1` hace el entrenamiento muy lento y el gradiente muy ruidoso.
- No hay callbacks (`EarlyStopping`, `ModelCheckpoint`) ni semilla aleatoria fija (resultados no reproducibles entre corridas).
- No guarda ninguna métrica, gráfica ni matriz de confusión — no queda registro de qué tan bien generalizó cada corrida.
- Sobrescribe siempre el mismo nombre de archivo; las versiones 2 y 3 que existen en el repo no tienen origen documentado.
- No guarda el mapeo de `class_indices` a ningún archivo.

## 5. Etapa 3 — Predicción en vivo (`prediccion.py`)

**Propósito:** usar el modelo ya entrenado para reconocer, en tiempo real desde la cámara, qué vocal está haciendo la mano.

**Paso a paso de lo que hace el script:**

1. Carga el modelo (`load_model('modeloVocales.keras')`) y sus pesos (`load_weights('pesosVocales.weights.h5')`).
2. Lee los nombres de las carpetas de `Validacion/` con `os.listdir` para saber qué letra corresponde a cada índice (0→primera carpeta en orden del sistema de archivos, etc.). **Esto asume que ese orden es el mismo que usó Keras al entrenar, lo cual no está garantizado.**
3. Abre la cámara y el detector de manos de MediaPipe, igual que en `lector.py`.
4. En cada fotograma, si detecta una mano:
   - Calcula el mismo recuadro de 200×200 alrededor del punto 9 (centro de la palma), pero esta vez **sí limitando el recorte a los bordes del frame** con `max(0, ...)` / `min(frame.shape, ...)` (esta parte ya está corregida respecto a `lector.py`).
   - Convierte el recorte a array (`img_to_array`) y le agrega una dimensión de "batch" (`np.expand_dims`).
   - **⚠️ Bug crítico:** el recorte se pasa al modelo **sin redimensionarlo a 200×200 y sin dividirlo por 255**. El modelo se entrenó esperando exactamente ese preprocesamiento (`ImageDataGenerator(rescale=1./255)` + `target_size=(200,200)`). Como el recuadro puede tener un tamaño distinto (si quedó recortado por los bordes) y los píxeles llegan en escala 0-255 en vez de 0-1, el modelo recibe un input que no se parece al que vio en entrenamiento. Esto probablemente es la causa principal de que las predicciones en vivo no sean confiables.
   - Llama a `cnn.predict(x)` y toma la clase con mayor probabilidad (`np.argmax`).
   - Según el índice devuelto (0 a 4), dibuja el recuadro y escribe sobre el video el nombre de la carpeta correspondiente (es decir, la letra). Si el índice no cae en 0-4 (lo cual no puede pasar con softmax de 5 clases, así que esta rama `else` en la práctica es inalcanzable), muestra "LETRA DESCONOCIDA".
5. Muestra el video con las anotaciones. Termina con ESC.

**Cómo se ejecuta:**

```
python prediccion.py
```

Requiere que ya exista `modeloVocales.keras` y `pesosVocales.weights.h5` (generados en la etapa 2), y que `Validacion/` siga teniendo las 5 subcarpetas (se usan solo para leer los nombres de las clases, no las imágenes).

**Problemas conocidos de este script** (ver sección 6 para la lista completa):
- El bug de preprocesamiento explicado arriba.
- No valida que el recorte no esté vacío (mano muy cerca del borde → recorte de 0 píxeles → error).
- No hay umbral de confianza mínimo: siempre anuncia una letra, aunque la probabilidad ganadora sea baja.
- Todo el ciclo de cámara + detección + predicción + dibujo está en un único bucle imperativo, sin separar la función de "predecir una imagen" del resto (necesario para poder exponerlo como servicio/API).

## 6. Lista completa de problemas y limitaciones detectados

### Bugs / correctitud
- **Preprocesamiento inconsistente entre entrenamiento y predicción** (`prediccion.py`): falta `resize` a 200×200 y `/255` antes de `predict`. Es el bug más importante a corregir.
- `lector.py`: el guardado de fotos está deshabilitado (comentado); tal como está, no genera dataset nuevo.
- `lector.py` y `prediccion.py`: si la mano está cerca del borde del frame, el recorte puede quedar vacío o (en `lector.py`) fuera de rango, y explotar.
- Ningún script valida `ret` al leer la cámara (si falla la lectura de un frame, `frame` es `None` y falla en `cv2.cvtColor`).
- El mapeo de índice de clase → letra depende de que `os.listdir` (en `prediccion.py`) devuelva las carpetas en el mismo orden que Keras usó al entrenar (`flow_from_directory`). Son dos fuentes de verdad independientes para lo mismo; no hay garantía de que coincidan siempre en todos los sistemas.

### Calidad del modelo / dataset
- Solo reconoce 5 vocales, no el alfabeto completo de lengua de señas.
- El dataset fue capturado (aparentemente) por una sola persona, con un fondo y condiciones de luz fijas → alto riesgo de que el modelo memorice el fondo/persona en vez de la forma de la mano, y generalice mal con otros usuarios.
- No hay clase "sin seña / mano fuera de cuadro" — el modelo siempre va a elegir una de las 5 vocales aunque no haya una seña válida.
- `batch_size = 1` en el entrenamiento es inusualmente bajo (entrenamiento lento y gradiente ruidoso).
- No hay medición de desempeño más allá del accuracy final que imprime Keras en consola: falta matriz de confusión, accuracy por clase, curva de aprendizaje (loss/accuracy por época) para diagnosticar sobreajuste.
- 3 versiones de modelo/pesos en el repo sin ningún registro (changelog, notas, métricas) de en qué se diferencian ni cuál es la mejor.
- No hay semilla aleatoria fija → cada entrenamiento da resultados distintos, sin forma de reproducir un resultado concreto.

### Higiene de proyecto
- `lector.py` importa librerías de otro proyecto (NLP) que no se usan: `Tokenizer`, `pad_sequences`, `Conv1D`, `MaxPooling1D`, `Embedding`, `Model`, `SparseCategoricalCrossentropy`, `train_test_split`, `pandas`, `matplotlib`.
- No hay `requirements.txt` ni versiones ancladas de `tensorflow`, `keras`, `mediapipe`, `opencv-python`, `numpy` — riesgo de incompatibilidades al reinstalar el entorno.
- Carpetas `Entrena/` y `Valida/` duplican la estructura de `Entrenamiento/`/`Validacion/` sin que ningún script las use hoy — no está documentado su propósito (¿backup? ¿dataset crudo antes de recortar?).
- No hay pruebas automatizadas.
- Todo el código de captura/predicción está escrito como script imperativo con `while(1)`, sin separar "lógica de inferencia" de "aplicación de cámara" — hace falta refactorizar antes de exponerlo como servicio.

## 7. Dependencias del proyecto

Ya están fijadas en `requirements.txt` (incluye tensorflow, opencv, mediapipe, numpy, y fastapi/uvicorn/python-multipart para la API). Instalar con:

```
pip install -r requirements.txt
```

**Importante:** el `.venv` incluido en la carpeta del proyecto está vacío (no tiene ningún paquete instalado, ni siquiera `pip`) — hay que crear uno nuevo o instalar las dependencias en el entorno donde efectivamente se corran los scripts.

`lector.py` importa `pandas`, `scikit-learn` y `matplotlib`, pero **no se usan** — quedaron de otro proyecto y se pueden quitar de las importaciones (no hace falta instalarlas para que el proyecto funcione).

## 8. Módulo de inferencia (`inferencia.py`) y corrección del bug

Se extrajo toda la lógica de detección de mano + predicción a `inferencia.py`, reutilizada tanto por `prediccion.py` (demo de escritorio) como por `api.py`. Esto además corrigió el bug crítico descrito en la sección 5: ahora el recorte de la mano se redimensiona a 200×200 y se reescala a [0,1] (`_preprocesar`) antes de pasarlo al modelo — igual que en el entrenamiento.

Cambios clave respecto a la versión anterior:

- **`predecir_desde_frame(frame_bgr)`**: función central. Recibe un frame de OpenCV (BGR) y devuelve `{mano_detectada, letra, confianza, caja}`. No abre cámara ni ventanas — por eso se puede llamar tanto desde un bucle de video como desde un endpoint HTTP.
- **Umbral de confianza** (`UMBRAL_CONFIANZA = 0.6`): si la clase ganadora no supera ese umbral, `letra` viene en `None` en vez de forzar una respuesta. Ajustable en `inferencia.py`.
- **`CLASES = ["A", "E", "I", "O", "U"]`** queda fijo en el código (ya no depende de leer `Validacion/` con `os.listdir`, que era una fuente de verdad frágil). Coincide con el orden alfabético que usa `flow_from_directory` al entrenar. **Si en el futuro se reentrena con más clases o se reordenan las carpetas, hay que actualizar esta lista a mano.**
- Manejo de recorte vacío (mano muy cerca del borde): devuelve `letra: None` en vez de crashear.

`prediccion.py` quedó simplificado: solo abre la cámara, llama a `predecir_desde_frame`, y dibuja el recuadro + texto con el resultado.

## 9. API (`api.py`)

Construida con **FastAPI**, expone la misma lógica de `inferencia.py` como servicio HTTP.

**Cómo correrla:**

```
uvicorn api:app --reload
```

Documentación interactiva (Swagger) una vez levantada: `http://127.0.0.1:8000/docs`

**Endpoints:**

- `GET /salud` → `{"estado": "ok"}`, para chequear que la API está arriba.
- `GET /clases` → `{"clases": ["A", "E", "I", "O", "U"]}`.
- `POST /predecir` → recibe una imagen como archivo (`multipart/form-data`, campo `imagen`) y devuelve:
  ```json
  {
    "mano_detectada": true,
    "letra": "A",
    "confianza": 0.87,
    "caja": [120, 80, 320, 280]
  }
  ```
  Si no se detecta una mano, `mano_detectada` es `false` y el resto viene en `null`. Si se detecta una mano pero la confianza no supera el umbral, `letra` y el resto quedan en `null` salvo `caja`.

**Ejemplo de request con curl:**

```
curl -X POST "http://127.0.0.1:8000/predecir" -F "imagen=@foto_mano.jpg"
```

**Limitación actual de este diseño:** la API recibe una imagen completa por request (no un stream de video) y hace la detección de mano del lado del servidor con MediaPipe en cada llamada — no hay seguimiento (tracking) entre frames como en el script de escritorio. Para un caso de uso de video en vivo contra la API, el cliente tendría que ir mandando frames sueltos (por ejemplo, uno cada X milisegundos) o habría que agregar un endpoint de WebSocket — no implementado todavía.

## 10. Próximos pasos pendientes

1. ~~Corregir el bug de preprocesamiento en `prediccion.py`.~~ ✅ Hecho (ahora vive en `inferencia.py`).
2. ~~Extraer la lógica de inferencia a un módulo reusable.~~ ✅ Hecho (`inferencia.py`).
3. ~~Construir la API con FastAPI.~~ ✅ Hecho (`api.py`).
4. ~~Probar de punta a punta con el modelo real.~~ ✅ Hecho. Se instaló `requirements.txt` en `.venv`, se corrigió un bug adicional que impedía cargar el modelo (ver sección 11), y se confirmó que `prediccion.py` y `api.py` funcionan de punta a punta (la API responde 200 OK en `/salud`, `/clases` y `/predecir`). Se detectó además un problema serio de precisión real del pipeline completo (ver sección 11) que hay que resolver antes de dar el proyecto por confiable.
5. Si se quiere quitar la dependencia de `flow_from_directory`/orden alfabético para el mapeo de clases, modificar `entrenamiento.py` para que guarde `class_indices` en un `.json` junto al modelo, y que `inferencia.py` lo lea en vez de tener la lista `CLASES` hardcodeada.
6. Ampliar el dataset (más letras del alfabeto, más personas, fondos e iluminación) para que el modelo generalice mejor — sigue siendo la limitación más importante para que esto sea un lector de lengua de señas real y no solo de 5 vocales.
7. Agregar métricas de evaluación al entrenamiento (matriz de confusión, accuracy por clase) y versionado de modelos (qué hiperparámetros/dataset generó cada `.keras`).
8. Limpiar las importaciones no usadas en `lector.py` y decidir qué hacer con las carpetas `Entrena/`/`Valida/` (documentar su propósito o eliminarlas). *(Ya documentado su propósito: son las fotos crudas fuente, ver sección 17. Siguen sin eliminarse.)*
9. ~~Corregir el desajuste entre el recorte de la predicción en vivo y el encuadre de entrenamiento.~~ ✅ Hecho, en dos etapas — ver secciones 12, 15 y 16. El encuadre ahora se calcula en un único módulo (`recorte.py`) que comparten entrenamiento e inferencia.
10. **Prioritario:** probar `prediccion.py` con la cámara en vivo usando el modelo actual (`modeloVocales_mano.keras`, entrenado con los recortes escalados a la mano). Es la única prueba que realmente vale: todas las mediciones offline son optimistas por el punto 11.
11. **Capturar un set de validación nuevo con la webcam** (otro momento, otra iluminación, idealmente otra persona) y medirlo con `evaluar_pipeline.py`. Todo el dataset actual es de una sola persona, un solo fondo y una sola sesión, así que ni el reparto por bloques alcanza para estimar el desempeño real.
12. Aplicar el mismo criterio de "entrenar con exactamente el recorte que ve la cámara" (secciones 12 y 15) si en el futuro se amplía el dataset con más letras o personas — el proyecto falló tres veces por variantes de este mismo error.
13. Revisar si las señas de A y E del dataset son distinguibles: en los recortes, A (puño con pulgar al costado) y E (puño con dedos doblados) se ven muy parecidas, y eran una de las confusiones frecuentes.

## 11. Resultados de la prueba de punta a punta (esta sesión)

### Bug encontrado y corregido: el modelo no cargaba

Al intentar cargar `modeloVocales.keras` con la versión de Keras instalada (3.3.3, vía `tensorflow==2.16.1`), `load_model(...)` fallaba con:

```
ValueError: A total of 1 objects could not be loaded. ...
'Unable to synchronously open object (bad object header version number)'
```

Se verificó que el archivo **no está corrupto ni mal descargado** (el hash SHA-256 del archivo coincide exactamente con el que espera Git LFS). El problema es que el estado interno del optimizador Adam, guardado dentro del `.keras`, quedó en un formato HDF5 que esta versión de Keras no puede leer (probablemente se guardó con una versión distinta de Keras/h5py a la que hay instalada ahora).

**Solución aplicada** en `inferencia.py`: cargar el modelo con `load_model(RUTA_MODELO, compile=False)`. El optimizador solo hace falta para *reentrenar*, no para predecir, así que omitirlo no afecta la inferencia. Los pesos de las capas (lo único que importa para predecir) se cargan igual con `load_weights(RUTA_PESOS)` justo después.

**Nota para el futuro:** si se reentrena el modelo (`entrenamiento.py`) con el entorno actual (TF 2.16.1 / Keras 3.3.3), el `.keras` resultante debería guardarse sin este problema, porque se generaría con la misma versión que lo lee.

### Hallazgo de precisión: el pipeline completo predice mal, pero el modelo en sí es bueno

Se hicieron dos pruebas separadas contra las fotos de `Validacion/` (75 fotos, 15 por vocal):

1. **Modelo solo** (redimensionar la foto completa a 200×200 y `/255`, sin pasar por MediaPipe): **92% de aciertos (69/75)**. Esto confirma que la CNN entrenada generaliza razonablemente bien sobre datos del mismo estilo que el dataset.
2. **Pipeline completo** (`predecir_desde_frame`, el mismo camino que usan `prediccion.py` y `api.py`: detectar la mano con MediaPipe sobre la imagen, recortar 200×200 alrededor del landmark 9, y recién ahí predecir): **14% de aciertos (7/50)**, con muchísimos casos donde ni siquiera se detectó una mano.

**Causa probable:** las fotos de `Validacion`/`Entrenamiento` ya son recortes cerrados de la mano (400×400, la mano ocupa casi todo el cuadro). Al pasarle esa foto a `predecir_desde_frame`, MediaPipe tiene que detectar landmarks de mano *dentro de un recorte que ya es un recorte* — a veces no encuentra mano (foto ya muy pegada, sin contexto de brazo/fondo) y cuando la encuentra, el nuevo recorte de 200×200 alrededor del landmark 9 termina siendo un encuadre distinto al que vio el modelo en entrenamiento (más zoom, descentrado, etc.), lo que degrada la predicción.

**Importante — esta prueba tiene una limitación:** `predecir_desde_frame` está pensado para recibir el *frame completo* de la cámara (mano chica dentro de un fondo, como hace `lector.py` al capturar), no una foto ya recortada como las de `Validacion`. No se pudo probar con frames de cámara reales en esta sesión (no hay cámara/dataset de frames completos disponible), así que el 14% **no es necesariamente la precisión real en vivo** — pero sí expone que el recorte en vivo puede no reproducir fielmente el encuadre del dataset de entrenamiento, y que MediaPipe puede fallar en detectar la mano cuando esta ocupa gran parte del cuadro (algo que puede pasar también en uso real, si el usuario acerca mucho la mano a la cámara).

**Recomendación (superada, ver sección 12):** antes de confiar en el proyecto para uso real, probar `prediccion.py` con la cámara en vivo. Esto se hizo en una sesión posterior y confirmó el problema — ver sección 12.

## 12. Prueba con cámara real y primera causa raíz (sesión posterior)

> ⚠️ **Leer junto con la sección 15.** El 91% que se reporta al final de esta sección resultó ser **inválido**: el dataset con el que se midió estaba mal generado. La sección 15 explica por qué y cómo se corrigió. Se conserva esta sección porque el diagnóstico del fondo segmentado sí es correcto y útil.

Se probó `prediccion.py` con la cámara real del usuario. Resultados observados:

- Predicción muy mala en general (coincide con el 14% medido en la sección 11).
- Funcionaba notablemente mejor con **fondo blanco** que con un fondo cualquiera (habitación, cara, pelo).
- La letra **I** casi nunca se reconocía: se imprimieron las probabilidades de las 5 clases en cada frame (`inferencia.py` ahora devuelve también `probabilidades`, un dict `{letra: probabilidad}`, además de `letra`/`confianza`) y con la mano haciendo la seña de "I" el modelo nunca le daba más de ~27% a esa clase — la confundía con U y con A.
- El tamaño de fuente en el video era demasiado grande (`prediccion.py`, `cv2.putText`) — corregido (de escala 1.3 a 0.7, thickness de 2 a 1).

**Primera causa encontrada: el dataset de entrenamiento tiene el fondo eliminado.** Comparando `Entrenamiento/I/00.jpg` (la foto que usó el entrenamiento: mano recortada sobre **fondo negro**) contra `Entrena/I/I1.jpg` (carpeta sin usar: la **misma foto sin procesar**, con fondo real — pared/marco de puerta), se confirmó que `Entrenamiento/`/`Validacion/` son versiones de `Entrena/`/`Valida/` con el fondo removido por alguna herramienta externa, sin registro de cuál. El modelo nunca vio un fondo real durante el entrenamiento. Esto explica por qué andaba mejor con fondo blanco (más parecido, por uniformidad, al fondo negro de entrenamiento) y por qué fallaba con fondos con textura.

Además se descubrió que `Entrena/` y `Valida/` son **archivos idénticos** (mismo hash MD5), no un split real de entrenamiento/validación — solo hay 99 fotos crudas por clase (495 en total) para trabajar.

**Primer intento de arreglo (parcial):** se reentrenó (`entrenamiento_fondo_real.py`, descartado luego, ver más abajo) usando un split real 80/20 de `Entrena/` (fondo real, foto completa redimensionada a 200×200, igual que hace `entrenamiento.py`). Resultado sobre el split de validación: **92.9% accuracy** entrenando, pero evaluando el *pipeline completo* (MediaPipe + recorte de 200×200 alrededor del landmark 9 + predicción) sobre esas mismas fotos: apenas **30% de aciertos**, aunque ya sin casos de "mano no detectada" (antes fallaba mucho también por eso). Fondo real arreglado, pero la precisión en vivo seguía siendo mala.

**Segunda causa, la dominante: el recorte en vivo no captura la mano completa.** Se midió el tamaño real de la mano en una foto cruda: ~271×353 píxeles dentro de un frame de 442×594. El recorte de `_recorte_mano` es una ventana fija de 200×200 centrada en un solo punto (landmark 9, el centro de la palma) — mucho más chica que la mano completa, así que corta dedos y bordes. Mientras que el entrenamiento usaba la **foto entera** (mano completa, con margen) redimensionada a 200×200, la predicción en vivo le muestra al modelo solo un recorte parcial y con mucho más zoom relativo. Son dos encuadres distintos de la misma seña — el modelo nunca aprendió a reconocer ese recorte parcial.

**Arreglo definitivo: entrenar con el mismo recorte que ve la cámara en vivo.** Se armó un pipeline nuevo:

1. `preparar_recortes.py`: toma las fotos crudas de `Entrena/` (fondo real) y les aplica el **mismo recorte** que hace `_recorte_mano` en `inferencia.py` (detecta la mano con MediaPipe, recorta 200×200 alrededor del landmark 9, redimensiona) → `RecortesCrudos/{A,E,I,O,U}/`. De 495 fotos, se generaron 488 recortes (7 sin detección de mano).
2. `repartir_recortes.py`: separa `RecortesCrudos/` en `EntrenamientoRecorte/` (80%) y `ValidacionRecorte/` (20%), por clase.
3. `entrenamiento.py`: igual que `entrenamiento.py`, pero apuntando a `EntrenamientoRecorte/`/`ValidacionRecorte/`. También corrige dos problemas de compatibilidad con la versión actual de TensorFlow/Keras que no tienen que ver con el dataset (ver sección 13) y sube `batch_size` de 1 a 16 (con solo ~400 imágenes, batch_size=1 tardaba más de una hora; con 16 entrena en minutos y el gradiente es menos ruidoso). Genera `modeloVocales_recorte.keras` / `pesosVocales_recorte.weights.h5`.
4. `evaluar_pipeline.py`: carga ese modelo y mide el accuracy sobre `ValidacionRecorte/`.

**Resultado:** accuracy del pipeline completo pasó de 14% (modelo original) → 30% (fondo real, sin arreglar encuadre) → **91%** (fondo real + mismo recorte que la cámara). Por clase: A 90%, E 100%, I 100%, O 74%, U 90% (O se confunde con E en varios casos; posible confusión visual real entre esas dos señas, o falta de ejemplos).

`inferencia.py` y `api.py` ya se actualizaron para usar `modeloVocales_recorte.keras`/`pesosVocales_recorte.weights.h5` (no requirió cambios en `api.py`, que no hardcodea la ruta del modelo). El modelo original (`modeloVocales.keras`) queda sin usar, disponible para comparación.

## 15. CORRECCIÓN IMPORTANTE: el 91% de la sección 12 era inválido

Al probar el modelo de la sección 12 con la cámara real, **predecía "E" para cualquier seña** (con 95-99% de confianza). Investigando por qué, se descubrió que el dataset con el que se lo entrenó era basura:

**Las fotos de `Entrena/` no son de webcam, son de cámara de fotos: miden entre 3968×2976 y 5120×3840 píxeles.** El recorte de `_recorte_mano` era una ventana **fija de 200×200 píxeles**. Sobre una foto de 4000 px de ancho, 200×200 px no es una mano: es un parche de piel de la palma. Se verificó abriendo `RecortesCrudos/A/A1.jpg`, que era literalmente un rectángulo de piel marrón, sin forma de mano.

Es decir: el modelo se entrenó para clasificar **parches de piel**, y el 91% medía su capacidad de distinguir texturas e iluminación entre fotos — no de reconocer señas. En vivo, la webcam (640×480) sí manda una mano entera dentro de esos 200×200, que no se parece a nada de lo que el modelo vio, y por eso colapsaba a una sola clase.

**Lección de fondo:** el error se repitió tres veces con distinta forma (fondo segmentado, foto completa vs. recorte, recorte fijo vs. escalado). Siempre era la misma causa: **la imagen que ve el entrenamiento no es la misma que ve la predicción.** Y el accuracy de validación nunca lo detectó, porque validación sufría exactamente la misma distorsión que entrenamiento.

### Arreglo: el recorte se escala al tamaño de la mano (`recorte.py`)

Se creó `recorte.py`, que es ahora **la única fuente de verdad** del encuadre. Su función `recortar_mano`:

1. Calcula la caja que envuelve **los 21 landmarks** de la mano que detecta MediaPipe (no un solo punto).
2. Le agrega un margen del 25% por lado y la hace **cuadrada** (así al redimensionar la seña no se deforma).
3. Si la mano toca el borde del frame, **rellena con negro** en vez de recortar el cuadrado (mantiene la proporción).
4. Redimensiona a 200×200.

Como el recorte se calcula a partir del tamaño de la mano, el resultado se ve igual en una foto de 5120 px o en un frame de webcam de 640 px, y a cualquier distancia de la cámara.

Lo importante es que **`inferencia.py` y `preparar_recortes.py` importan esa misma función**, en vez de tener cada uno su copia del cálculo. Así el desalineamiento que causó todo este problema no puede volver a ocurrir por descuido: si se cambia el recorte, cambia en los dos lados a la vez.

Se verificó visualmente que los recortes nuevos sí son manos completas antes de reentrenar (paso que no se había hecho antes, y que habría detectado el problema de entrada).

### Segunda corrección: el reparto entrenamiento/validación era optimista

El reparto original de `repartir_recortes.py` era **aleatorio**. No había ninguna foto repetida entre las dos carpetas (se verificó: 0 archivos compartidos en las 5 vocales), así que el principio de "entrenar y validar con fotos distintas" se cumplía. Pero las fotos del dataset parecen tomadas en ráfaga, y se midió que **dos fotos consecutivas difieren en promedio 5.1 (escala 0-255) mientras que dos fotos alejadas difieren 11.6**: las consecutivas son más del doble de parecidas.

Con reparto aleatorio, `A5.jpg` podía quedar en entrenamiento y `A6.jpg` —casi su gemela, mismo instante, misma luz— en validación. Son archivos distintos, pero al modelo no le hace falta *generalizar* para acertarlas: le alcanza con reconocer esa toma puntual. Eso infla el número: con reparto aleatorio el entrenamiento llegaba a **val_accuracy 1.0000 (100%)**, un valor claramente irreal para un dataset de 99 fotos por clase.

`repartir_recortes.py` ahora reparte **por bloques**: ordena las fotos por su número (orden en que se tomaron) y manda el tramo final completo (fotos 80 a 99 de cada vocal) a validación. Así ninguna foto de validación es gemela de una de entrenamiento, y el accuracy resultante es honesto.

**Resultado del reparto por bloques: siguió dando val_accuracy 1.0000 (100%).** Se investigó por qué, midiendo cuán parecida es cada foto de validación a su foto *más parecida* del entrenamiento (diferencia media de píxeles, imágenes reducidas a 64×64):

| Vocal | Validación → su vecina más parecida de entrenamiento | Entrenamiento → su vecina más parecida (entre ellas) |
|---|---|---|
| A | **1.0** | 2.6 |
| U | 11.7 | 3.5 |

En la vocal A, cada foto de validación está **más cerca** de alguna foto de entrenamiento (1.0) de lo que están las fotos de entrenamiento entre sí (2.6): el bloque final son prácticamente duplicados de fotos que el modelo ya vio, así que separarlas no separó nada. En U sí funcionó (11.7 vs 3.5).

**Conclusión: con este dataset ningún reparto puede dar una medición honesta.** Las 99 fotos por vocal son una sola sesión, una sola persona, un solo fondo y una sola iluminación — para varias clases son casi la misma foto repetida. Un 100% de validación no dice nada sobre el desempeño real; solo dice que el modelo memorizó bien esa sesión.

**La única medición válida es probar con la cámara en vivo, o capturar un set nuevo de fotos** (otro momento, otra luz, idealmente otra persona) y evaluarlo con `evaluar_pipeline.py`. Ese es el paso pendiente más importante del proyecto (sección 10, puntos 10 y 11).

## 16. Otro bug de compatibilidad (Keras 3): el generador no se reinicia solo entre épocas

Al reentrenar con la versión actual de TensorFlow/Keras (2.16.1 / 3.3.3), `cnn.fit(...)` con un generador de `ImageDataGenerator().flow_from_directory(...)` fallaba en la **segunda** época con:

```
UserWarning: Your input ran out of data; interrupting training...
AttributeError: 'NoneType' object has no attribute 'items'
```

En versiones viejas de Keras ese generador se reiniciaba solo al agotarse, por eso el `entrenamiento.py` original funcionaba en su momento. En Keras 3, el adaptador que envuelve estos generadores clásicos no lo reinicia automáticamente entre épocas. **Arreglo** (ya aplicado en `entrenamiento.py`): envolver el generador en un `tf.data.Dataset.from_generator(...).repeat()` antes de pasarlo a `fit()`.

También apareció un `UnicodeEncodeError` al redirigir la salida de Keras a un archivo de log en Windows (la barra de progreso de Keras 3 usa caracteres Unicode que la consola en `cp1252` no puede escribir). Se resolvió corriendo el entrenamiento con las variables de entorno `PYTHONIOENCODING=utf-8` y `PYTHONUTF8=1`.

## 17. Estado de los datasets y carpetas

- `Entrena/`: **fuente de verdad**. 99 fotos crudas por vocal (fondo real, cámara de fotos, resolución alta). Todo lo demás se deriva de acá. Está versionada en git (con LFS).
- `RecortesCrudos/`: recortes de `Entrena/` generados por `preparar_recortes.py`, con el mismo encuadre que usa la cámara en vivo. **Derivada** — se regenera cuando haga falta.
- `EntrenamientoRecorte/` / `ValidacionRecorte/`: reparto 80/20 por bloques de `RecortesCrudos/`, hecho por `repartir_recortes.py`. **Derivada.**

## 18. Limpieza de la carpeta del proyecto

El proyecto tenía ~4.4 GB con mucho material redundante o superado. Se borró lo siguiente (todo estaba subido a GitHub —`origin/main` coincidía con el commit local— así que es recuperable con `git checkout` + `git lfs pull`):

| Borrado | Tamaño | Motivo |
|---|---|---|
| `Valida/` | 566 MB | Copia **exacta** de `Entrena/` (mismo hash MD5 archivo por archivo) |
| `Entrenamiento/` + `Validacion/` | 545 MB | Dataset con el fondo borrado (fondo negro) que causó el bug original; ningún script lo usa |
| `modeloVocales.keras` + `pesosVocales.weights.h5` | 1.2 GB | Modelo original, superado por `modeloVocales_mano.keras` |
| `modeloVocales2/3.keras`, `pesosVocales.weights2/3.h5` | 134 bytes c/u | Punteros LFS que nunca se descargaron: archivos sin contenido y sin documentación de en qué se diferenciaban |
| `RecortesCrudos/`, `EntrenamientoRecorte/`, `ValidacionRecorte/` | 11 MB | Derivadas, se regeneran con los scripts |
| Logs `.train_*.log`, `.prep2.log`, etc. | pocos KB | Diagnóstico temporal |

Además se **unificaron los dos scripts de entrenamiento**, que hacían lo mismo: se borró el `entrenamiento.py` original (entrenaba con el dataset de fondo negro, que ya no existe) y `entrenamiento_recorte.py` pasó a llamarse `entrenamiento.py`. Ahora hay un solo script de entrenamiento.

Quedó en **1.7 GB**: `Entrena/` (566 MB) + el par modelo/pesos en uso (1.2 GB).
