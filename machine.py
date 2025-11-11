import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Datos DATASET
# --- NOTE: One very large outlier in the 'precios' array: 160010000
metros = np.array([[50],[60],[80],[100],[120],[150]])
precios = np.array([[50000], [65000], [85000], [120000], [135000], [160000]])

# 2. Dividir y Entrenar el Modelo
x_train, x_test, y_train, y_test = train_test_split(
    metros, precios, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

# 3. Predicción del Nuevo Valor
try:
    # NOTE: Since I cannot run input(), I will use a dummy value for the prediction:
    # ntr = int(input("Digite el valor del metro cuadrado a predecir: "))
    ntr = 80  # Dummy value for demonstration
    nuevo_metro = np.array([[ntr]])

    # Generamos la predicción
    prediccion = model.predict(nuevo_metro)
    
    # 4. Mostrar el Resultado Formateado y Conciso
    precio_formateado = f"{prediccion[0][0]:,.2f}"

    # Calculate R-squared score
    score = model.score(x_test, y_test) 
    
    print(f'\nScore (R²): {score:.2f}')
    print(f"Predicción del modelo:")
    print(f"Para {ntr} m² el precio estimado es: ${precio_formateado}")

except ValueError:
    print("Error: Por favor, ingrese un número entero válido.")

# ---
# 5. Graficar (Correcciones Aplicadas Aquí 👇)
# ------------------------------------------------
# 1. Fixed 'np.linpace' to 'np.linspace'
# 2. Fixed 'modelo.predict' to 'model.predict'

# Generar 100 puntos espaciados entre el mínimo y el máximo de 'metros'
x_linea = np.linspace(min(metros), max(metros), 100).reshape(-1, 1)

# Usar el objeto 'model' correctamente para predecir
y_linea = model.predict(x_linea)

plt.figure(figsize=(8, 6))
# Gráfico de dispersión de los datos reales
plt.scatter(metros, precios, color='red', label="Datos reales (incluye outlier)")
# Gráfico de la línea de regresión (predicción del modelo)
plt.plot(x_linea, y_linea, color='blue', label="Línea del modelo de regresión")

plt.xlabel("Metros Cuadrados")
plt.ylabel("Precio ($us.)")
plt.title("Regresión Lineal: Metros vs. Precio")
plt.legend()
plt.grid(True)
plt.show()