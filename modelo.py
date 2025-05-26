# modelo.py  – Versión mejorada
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ----------------------------------------------------------
# 0. Configuración general
# ----------------------------------------------------------
os.makedirs("graficos", exist_ok=True)          # carpeta para las imágenes
RANDOM_STATE = 42                               # reproducibilidad

# ----------------------------------------------------------
# 1. Carga del dataset
# ----------------------------------------------------------
data = fetch_california_housing()
df   = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target                     # precio de la vivienda

# ----------------------------------------------------------
# 2. Regresión lineal simple (AveRooms) con filtrado de outliers
# ----------------------------------------------------------
#   – Eliminar valores extremos de AveRooms (> 20)
df_rl = df[df["AveRooms"] < 20].copy()

X_simple = df_rl[["AveRooms"]]
y_simple = df_rl["target"]

lin_reg = LinearRegression()
lin_reg.fit(X_simple, y_simple)
pred_simple = lin_reg.predict(X_simple)

#  Métricas
mse_rl = mean_squared_error(y_simple, pred_simple)
r2_rl  = r2_score(y_simple, pred_simple)
print(f"\nRegresión Lineal  |  MSE = {mse_rl:.3f}  |  R² = {r2_rl:.3f}")

#  Gráfico
plt.figure(figsize=(8, 5))
plt.scatter(X_simple, y_simple, alpha=0.3, label="Valores reales")
plt.plot(X_simple, pred_simple, color="red", label="Predicción")
plt.xlabel("Promedio de habitaciones (AveRooms)")
plt.ylabel("Precio medio de vivienda")
plt.title("Regresión Lineal Simple (sin outliers)")
plt.legend()
plt.tight_layout()
plt.savefig("graficos/regresion_lineal.png")
plt.close()

# ----------------------------------------------------------
# 3. Red neuronal multivariable con Keras
# ----------------------------------------------------------
X_full = df.drop(columns="target")
y_full = df["target"]

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_full, test_size=0.2, random_state=RANDOM_STATE
)

#  Arquitectura mejorada
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

#   – Early Stopping para evitar sobre-ajuste
early_stop = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

#  Predicciones y métricas
y_pred = model.predict(X_test).flatten()
mse_nn = mean_squared_error(y_test, y_pred)
r2_nn  = r2_score(y_test, y_pred)
print(f"Red Neuronal      |  MSE = {mse_nn:.3f}  |  R² = {r2_nn:.3f}")

#  Gráfico con línea y = x
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3, label="Predicción")
plt.plot([0, 5], [0, 5], "r--", label="Ideal (y = x)")
plt.xlabel("Valor Real")
plt.ylabel("Predicción")
plt.title("Red Neuronal – Predicción de Precios")
plt.legend()
plt.tight_layout()
plt.savefig("graficos/red_neuronal.png")
plt.close()

# ----------------------------------------------------------
# 4. Ejemplo de estructuras de control y datos
# ----------------------------------------------------------
resultados = []
for real, pred in zip(y_test[:10], y_pred[:10]):
    error = abs(real - pred)
    resultados.append({
        "real": round(real, 2),
        "predicho": round(pred, 2),
        "error": round(error, 2),
        "comentario": "Preciso" if error < 0.5 else "Mejorable"
    })

print("\nPrimeros 10 resultados:")
for r in resultados:
    print(r)
