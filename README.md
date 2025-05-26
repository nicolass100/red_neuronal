#  Construcción de una Red Neuronal para Predecir Precios de Viviendas 

Este proyecto tiene como finalidad aplicar conocimientos de *Machine Learning* mediante el uso de **regresión lineal** y **redes neuronales artificiales** para predecir precios de viviendas a partir de características del entorno. Se desarrolló en Python utilizando bibliotecas como **TensorFlow/Keras**, **Scikit-learn**, **Pandas** y **Matplotlib**.

---

##  Objetivo

Desarrollar un modelo de predicción de precios de viviendas que:
- Utilice regresión lineal simple con una variable (ej. número de habitaciones)
- Implemente una red neuronal básica utilizando TensorFlow/Keras con múltiples variables
- Permita visualizar y comparar los resultados mediante gráficos
- Se gestione de forma colaborativa y segura con control de versiones a través de GitHub

---

##  Metodología

1. **Investigación teórica**: Se revisaron los conceptos de redes neuronales, regresión lineal y herramientas como TensorFlow y GitHub.
2. **Preparación del entorno**:
   - Instalación de paquetes necesarios (`tensorflow`, `keras`, `scikit-learn`, etc.)
   - Inicialización de repositorio GitHub y documentación en `README.md`
3. **Carga y preprocesamiento del dataset**:
   - Se usó el conjunto de datos `California Housing` de Scikit-learn
   - Normalización de variables con `StandardScaler`
4. **Modelado**:
   - Regresión lineal simple con `LinearRegression` de Scikit-learn
   - Red neuronal con varias capas densas usando `Sequential` de Keras
5. **Evaluación de resultados**:
   - Visualización de predicciones frente a valores reales
   - Análisis de precisión y comportamiento del modelo

---

##  Resultados Obtenidos

### Regresión Lineal:
- Se observó una relación lineal aceptable entre la variable "número promedio de habitaciones" y el precio de la vivienda.
- El modelo es simple pero con limitaciones en precisión.

### Red Neuronal:
- La red neuronal mostró mejor capacidad predictiva al usar múltiples variables.
- Reducción significativa del error cuadrático medio comparado con la regresión simple.
- Gráficamente, las predicciones se ajustaron mejor a los valores reales.

---

##  Conclusiones

- Las redes neuronales, aunque más complejas, ofrecen mayor poder predictivo cuando se manejan múltiples variables.
- El uso de herramientas como GitHub permite un desarrollo organizado, colaborativo y controlado.
- Este proyecto refuerza el entendimiento práctico de técnicas de ML básicas y el flujo de trabajo completo: desde la carga de datos hasta la visualización final.

---

## Tecnologías utilizadas

- **Python 3.12**
- **TensorFlow / Keras**
- **Scikit-learn**
- **Pandas**
- **Matplotlib**
- **Git / GitHub**

---

##  Estructura del proyecto

