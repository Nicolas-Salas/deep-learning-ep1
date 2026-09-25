# Clasificación Curricular Automática con Redes Neuronales Multicapa (MLP)

**Evaluación Parcial N.° 1 — Fundamentos de Deep Learning (DLY0100)**
DUOC UC · Sección DEEP LEARNING_011V_OLS

| Integrante | Responsabilidad principal |
| :--- | :--- |
| Rahyzza Novoa | Funciones de activación y pérdida, arquitectura, evaluación en test y métricas |
| Nicolas Salas | Hiperparámetros de entrenamiento, early stopping, regularización y optimizadores |

---

## Descripción del problema

El proyecto aborda la clasificación automática de material educativo de matemáticas en cuatro categorías curriculares a partir de 22 variables numéricas que describen su estructura, contenido y diseño.

| Clase | Categoría | Proporción |
| :---: | :--- | :---: |
| 0 | Números | 30,22% |
| 1 | Álgebra | 26,97% |
| 2 | Geometría | 23,07% |
| 3 | Datos y Probabilidad | 19,75% |

Dado el desbalance moderado entre clases, se utiliza el **F1-score ponderado** como métrica principal de comparación.

---

## Estructura del repositorio

```
deep-learning-ep1/
├── data/
│   └── Dataset_Clasificacion_Curricular_100000_LIMPIO.csv   # 100.000 registros, 23 columnas
├── docs/
│   ├── EP1_DLY0100_Estudiante.pdf                           # Documento oficial de la evaluación
│   ├── Consigna_Parcial1_Clasificacion_Curricular.pdf       # Caso de estudio
│   └── presentacion/                                        # Presentación de apoyo (PPTX y PDF)
├── notebook/
│   ├── deep_learning_ev1.ipynb                              # Experimentos B1-B5 y modelo final
│   ├── Arquitectura, funciones y experimentos.ipynb         # Experimentos A1-A3 y evaluación en test
│   ├── modelo_final.keras
│   ├── modelo_final_pesos.npz
│   └── resultados_experimentos.csv
└── README.md
```

| Archivo | Contenido |
| :--- | :--- |
| `modelo_final.keras` | Modelo entrenado con la configuración seleccionada |
| `modelo_final_pesos.npz` | Pesos del modelo final en formato NumPy, independientes de la versión de Keras. Los utiliza el segundo notebook para la evaluación en test |
| `resultados_experimentos.csv` | Registro consolidado de todas las corridas del notebook principal con sus métricas |

---

## Requisitos

| Paquete | Versión utilizada |
| :--- | :--- |
| Python | 3.12 |
| TensorFlow / Keras | 2.21 |
| scikit-learn | ≥ 1.3 |
| pandas | ≥ 2.0 |
| NumPy | ≥ 1.26 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |

Todos los paquetes vienen preinstalados en Google Colab.

---

## Instrucciones de ejecución

El proyecto se compone de **dos notebooks que se ejecutan en orden**:

1. `deep_learning_ev1.ipynb`: experimentos de entrenamiento (B1-B5) y modelo final. Genera `modelo_final_pesos.npz`.
2. `Arquitectura, funciones y experimentos.ipynb`: experimentos de funciones y arquitectura (A1-A3) y evaluación final sobre test, cargando los pesos del modelo final.

### Opción A — Ejecución local (recomendada)

```bash
git clone https://github.com/Nicolas-Salas/deep-learning-ep1.git
cd deep-learning-ep1
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
cd notebook
jupyter notebook
```

Abrir y ejecutar completos, en este orden, `deep_learning_ev1.ipynb` y luego `Arquitectura, funciones y experimentos.ipynb`. Ambos detectan automáticamente el dataset en `../data/` y los pesos del modelo en la carpeta `notebook/`.

### Opción B — Google Colab

1. **Archivo → Abrir cuaderno → GitHub**, pegar la URL de este repositorio y seleccionar el notebook.
2. Activar GPU: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**.
3. En el panel **Archivos**, subir `Dataset_Clasificacion_Curricular_100000_LIMPIO.csv` (carpeta `data/`). Para el segundo notebook, subir también `modelo_final_pesos.npz` (carpeta `notebook/`).
4. **Entorno de ejecución → Ejecutar todas**.

> **Reproducibilidad:** la semilla fija garantiza resultados idénticos en un mismo equipo. En otro hardware (por ejemplo, GPU de Colab) pueden aparecer diferencias numéricas pequeñas respecto de los valores documentados.

> **Importante:** cada notebook debe ejecutarse completo y en orden desde un kernel reiniciado. Los experimentos se acumulan en un registro global; re-ejecutar celdas aisladas genera filas duplicadas en las tablas comparativas.

---

## Metodología

### Preprocesamiento
- Partición **70/15/15 estratificada** (entrenamiento / validación / test), preservando la proporción de clases. Ambos notebooks usan la misma partición.
- **StandardScaler** ajustado exclusivamente sobre entrenamiento y aplicado a validación y test, evitando fuga de información.
- El conjunto de test se reserva para una única evaluación final.

### Reproducibilidad
Semilla fija (`SEED = 42`) en `random`, `numpy` y `tensorflow`, restablecida antes de cada entrenamiento, de modo que la única diferencia entre corridas es el parámetro variado.

### Experimentos controlados

Cada experimento varía **un único parámetro** y mantiene fijo el resto.

| Bloque | Parámetro evaluado | Valores | Seleccionado |
| :--- | :--- | :--- | :---: |
| B1 | Tasa de aprendizaje | 0.1 · 0.01 · 0.001 · 0.0005 | **0.0005** |
| B2 | Tamaño de lote | 32 · 64 · 128 · 256 | **256** |
| B3 | Épocas efectivas | Corrida de 100 épocas sin parada | **Early Stopping (patience = 10)** |
| B4 | Regularización | Ninguna · Dropout 0.2 · Dropout 0.4 · L2 · BatchNorm · Dropout + BatchNorm | **Dropout 0.2** |
| B5 | Optimizador | Adam · RMSprop · SGD | **Adam** |
| A1 | Función de activación | ReLU · tanh · sigmoide · LeakyReLU, y red profunda de 5 capas | **ReLU** |
| A2 | Salida y pérdida | Softmax + Cross-Entropy · Sigmoide + MSE | **Softmax + Sparse Categorical Cross-Entropy** |
| A3 | Arquitectura | 64 · 64-32 · 128-64-32 · 256-128 | **64-32** |

### Criterios de selección
1. F1-score ponderado en validación.
2. Ante diferencias de F1 inferiores a 0,005: menor `val_loss_min` y menor brecha absoluta entre entrenamiento y validación.
3. Estabilidad de la curva de validación.
4. Costo computacional, solo como desempate.

---

## Resultados principales

| Hallazgo | Evidencia |
| :--- | :--- |
| Una tasa de aprendizaje de 0.1 impide la convergencia | Accuracy 30,22% (igual a la clase mayoritaria) y val_loss ≈ ln 4, equivalente a predicción uniforme |
| Adam amortigua la sensibilidad a la tasa en el rango 0.01–0.0005 | Diferencia de F1 de 0,0024 entre las tres tasas |
| Batch 256 mejora desempeño y costo | F1 0,9129 en 25,26 s, frente a 0,9100 en 149,93 s con batch 32 |
| El sobreajuste de base es leve | val_loss asciende de 0,238 a 0,245 tras la época 16 |
| El efecto de la regularización es acotado | Rango de F1 de 0,0019 entre las seis variantes; Dropout 0.2 obtiene el menor val_loss_min (0,2341) |
| El optimizador no altera el desempeño final | F1 entre 0,9132 y 0,9133 con los tres; RMSprop converge un 24,5% más rápido y SGD no alcanza a converger en 100 épocas |
| En una red de 2 capas, las activaciones rinden igual | Diferencia de F1 de 0,0024; en una red de 5 capas, sigmoide atenúa el gradiente 1.959 veces y converge en 56 épocas frente a 36 de ReLU |
| Softmax es la única salida con probabilidades válidas | Mismo F1 que Sigmoide + MSE, pero las salidas de Sigmoide suman entre 0,82 y 1,49 |
| Más capacidad no mejora el modelo | Con 22 veces más parámetros, el F1 sube solo 0,0009 |
| La confusión principal es Números ↔ Álgebra | 185 y 164 registros; con umbral de confianza de 0,80 la precisión sube a 96,7% cubriendo el 82,1% |

### Modelo final

MLP 22 → 64 (ReLU, Dropout 0.2) → 32 (ReLU, Dropout 0.2) → 4 (Softmax), 3.684 parámetros. Adam con learning rate 0.0005, batch 256, Sparse Categorical Cross-Entropy y Early Stopping (patience 10).

### Desempeño del modelo final (conjunto de test)

| Métrica | Valor |
| :--- | :---: |
| Accuracy | 0,9108 |
| Precision (weighted) | 0,9108 |
| Recall (weighted) | 0,9108 |
| F1-score (weighted) | 0,9108 |

Diferencia con validación: −0,0025. El detalle por categoría, la matriz de confusión y el cálculo manual de las métricas se encuentran en `Arquitectura, funciones y experimentos.ipynb`.

---

## Referencias

- TensorFlow. *Keras API reference.* https://www.tensorflow.org/api_docs/python/tf/keras
- Keras. *EarlyStopping callback.* https://keras.io/api/callbacks/early_stopping/
- scikit-learn. *Model evaluation: classification metrics.* https://scikit-learn.org/stable/modules/model_evaluation.html
- Chollet, F. (2021). *Deep Learning with Python* (2.ª ed.). Manning.
- Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR, 15, 1929–1958.
- Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.