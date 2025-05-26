# modelo.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Cargar dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# 2. REGRESIÓN LINEAL SIMPLE con AveRooms
X_simple = df[['AveRooms']]
y = df['target']

modelo_rl = LinearRegression()
modelo_rl.fit(X_simple, y)
pred_simple = modelo_rl.predict(X_simple)

# 3. Guardar gráfico de regresión lineal
plt.figure(figsize=(8,5))
plt.scatter(X_simple, y, alpha=0.3, label="Valores reales")
plt.plot(X_simple, pred_simple, color="red", label="Predicción")
plt.xlabel("Promedio de habitaciones (AveRooms)")
plt.ylabel("Precio medio de vivienda")
plt.title("Regresión Lineal Simple")
plt.legend()
plt.savefig("graficos/regresion_lineal.png")
plt.close()

# 4. RED NEURONAL MULTIVARIABLE

# Preparar datos
X = df.drop(columns='target')
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Construir modelo de red neuronal
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, validation_split=0.2, verbose=0)

# Predicciones
y_pred = model.predict(X_test).flatten()

# 5. Guardar gráfico de red neuronal
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("Valor Real")
plt.ylabel("Predicción")
plt.title("Red Neuronal - Predicción de Precios")
plt.savefig("graficos/red_neuronal.png")
plt.close()

# 6. Estructuras de control y datos
resultados = []
for real, pred in zip(y_test[:10], y_pred[:10]):
    error = abs(real - pred)
    resultado = {
        'real': round(real, 2),
        'predicho': round(pred, 2),
        'error': round(error, 2),
        'comentario': "Preciso" if error < 0.5 else "Mejorable"
    }
    resultados.append(resultado)

# Mostrar resultados
for r in resultados:
    print(r)
