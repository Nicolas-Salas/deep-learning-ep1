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
│   └── Consigna_Parcial1_Clasificacion_Curricular.pdf       # Caso de estudio
├── notebook/
│   └── deep_learning_ev1.ipynb                              # Notebook principal (informe)
└── README.md
```

Al ejecutar el notebook completo se generan además:

| Archivo | Contenido |
| :--- | :--- |
| `modelo_final.keras` | Modelo entrenado con la configuración seleccionada |
| `resultados_experimentos.csv` | Registro consolidado de todas las corridas con sus métricas |

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

### Opción A — Google Colab (recomendada)

1. Abrir [Google Colab](https://colab.research.google.com) → **Archivo → Abrir cuaderno → GitHub** y pegar la URL de este repositorio. Seleccionar `notebook/deep_learning_ev1.ipynb`.
2. Activar GPU: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU**. Sin GPU, la ejecución completa puede superar los 40 minutos.
3. Subir el dataset: en el panel lateral **Archivos**, cargar `Dataset_Clasificacion_Curricular_100000_LIMPIO.csv` desde la carpeta `data/` de este repositorio.
4. Ejecutar todo: **Entorno de ejecución → Ejecutar todas**.

### Opción B — Ejecución local

```bash
git clone https://github.com/Nicolas-Salas/deep-learning-ep1.git
cd deep-learning-ep1
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn jupyter
cd notebook
jupyter notebook deep_learning_ev1.ipynb
```

Ejecutar todas las celdas en orden. El notebook detecta automáticamente la ubicación del dataset en `../data/`, `data/` o el directorio actual.

> **Importante:** el notebook debe ejecutarse completo y en orden desde un kernel reiniciado. Los experimentos se acumulan en un registro global (`df_experimentos`); re-ejecutar celdas aisladas genera filas duplicadas en las tablas comparativas.

---

## Metodología

### Preprocesamiento
- Partición **70/15/15 estratificada** (entrenamiento / validación / test), preservando la proporción de clases.
- **StandardScaler** ajustado exclusivamente sobre entrenamiento y aplicado a validación y test, evitando fuga de información.
- El conjunto de test se reserva para una única evaluación final.

### Reproducibilidad
Semilla fija (`SEED = 42`) en `random`, `numpy` y `tensorflow`, reestablecida antes de cada entrenamiento. Todos los experimentos se ejecutan mediante una función común (`entrenar_modelo`), de modo que la única diferencia entre corridas es el parámetro variado.

### Experimentos controlados

Cada experimento varía **un único parámetro** y mantiene fijo el resto.

| Bloque | Parámetro evaluado | Valores | Seleccionado |
| :--- | :--- | :--- | :---: |
| B1 | Tasa de aprendizaje | 0.1 · 0.01 · 0.001 · 0.0005 | **0.0005** |
| B2 | Tamaño de lote | 32 · 64 · 128 · 256 | **256** |
| B3 | Épocas efectivas | Corrida de 100 épocas sin parada | **Early Stopping (patience = 10)** |
| B4 | Regularización | Ninguna · Dropout 0.2 · Dropout 0.4 · L2 · BatchNorm · Dropout + BatchNorm | **Dropout 0.2** |
| B5 | Optimizador | Adam · RMSprop · SGD | **Adam** |
| A1 | Función de activación | ReLU · tanh · sigmoide · LeakyReLU | [pendiente] |
| A2 | Arquitectura | 1 a 3 capas ocultas, distintos anchos | [pendiente] |

### Criterios de selección
1. F1-score ponderado en validación.
2. Ante diferencias de F1 inferiores a 0,005: menor brecha entre entrenamiento y validación y menor `val_loss_min`.
3. Estabilidad de la curva de validación.
4. Costo computacional, solo como desempate.

---

## Resultados principales

| Hallazgo | Evidencia |
| :--- | :--- |
| Una tasa de aprendizaje de 0.1 impide la convergencia | Accuracy 30,22% (igual a la clase mayoritaria) y val_loss ≈ ln 4, equivalente a predicción uniforme |
| Adam amortigua la sensibilidad a la tasa en el rango 0.01–0.0005 | Diferencia de F1 de 0,0024 entre las tres tasas |
| Batch 256 mejora desempeño y costo | F1 0,9129 en 24,69 s, frente a 0,9100 en 154,27 s con batch 32 |
| El sobreajuste de base es leve | val_loss asciende de 0,238 a 0,245 tras la época 16 |
| El efecto de la regularización es acotado | Rango de F1 de 0,0019 entre las seis variantes; Dropout 0.2 obtiene el menor val_loss_min (0,2341) |
| El optimizador no altera el desempeño final | F1 entre 0,9132 y 0,9133 con los tres; RMSprop converge un 24,5% más rápido y SGD no alcanza a converger en 100 épocas |

### Desempeño del modelo final (conjunto de test)

| Métrica | Valor |
| :--- | :---: |
| Accuracy | [pendiente] |
| Precision (weighted) | [pendiente] |
| Recall (weighted) | [pendiente] |
| F1-score (weighted) | [pendiente] |

---

## Referencias

- TensorFlow. *Keras API reference.* https://www.tensorflow.org/api_docs/python/tf/keras
- Keras. *EarlyStopping callback.* https://keras.io/api/callbacks/early_stopping/
- scikit-learn. *Model evaluation: classification metrics.* https://scikit-learn.org/stable/modules/model_evaluation.html
- Chollet, F. (2021). *Deep Learning with Python* (2.ª ed.). Manning.
- Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR, 15, 1929–1958.
- Kingma, D. P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.