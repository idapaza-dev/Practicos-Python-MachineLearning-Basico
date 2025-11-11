import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# ==============================================================================
# 1. DATOS MULTIVARIABLES
# ==============================================================================
print("--- 1. Datos del Modelo Multivariable ---")

# X: [Metros, Habitaciones, Baños, Ubicación_Exclusiva (1=Sí, 0=No)]
X_data = np.array([
    [50,  1, 1, 0],
    [60,  2, 1, 0],
    [80,  2, 2, 1],
    [100, 3, 2, 1],
    [120, 3, 3, 1],
    [150, 4, 3, 1]  # Outlier: 150m² con precio de 160M
])

# y: Precio (en USD)
y_precios = np.array([[50000], [65000], [85000], [120000], [135000], [160000]])

# ==============================================================================
# 2. ENTRENAMIENTO Y EVALUACIÓN
# ==============================================================================
# Dividir datos (33% para prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_precios, test_size=0.33, random_state=42
)

model_4var = LinearRegression()
model_4var.fit(X_train, y_train)

# Evaluación
score = model_4var.score(X_test, y_test) 
y_pred_test = model_4var.predict(X_test) # Predicciones para el gráfico

print(f"\n--- 2. Resultados del Entrenamiento ---")
print(f"Score R² (Evaluación): {score:.2f} (Bajo debido al outlier en el set de entrenamiento)")
print(f"Coeficientes [Metros, Hab, Baños, Ubicación]: {model_4var.coef_[0]}")


# ==============================================================================
# 3. PREDICCIÓN CON NUEVOS VALORES
# ==============================================================================
print("\n--- 3. Predicción Específica ---")

# Definir las 4 características de la nueva vivienda a predecir:
metros_new = 90
hab_new = 2
banos_new = 2
ubi_new = 1 
nuevo_X_predict = np.array([[metros_new, hab_new, banos_new, ubi_new]])

prediccion = model_4var.predict(nuevo_X_predict)
precio_formateado = f"{prediccion[0][0]:,.2f}"

print(f"Características: {metros_new}m², {hab_new} hab, {banos_new} baños, Ubicación Exclusiva.")
print(f"Precio Estimado: ${precio_formateado}")


# ==============================================================================
# 4. GRÁFICA: PRECIOS REALES VS. PRECIOS PREDICTOS
# ==============================================================================
print("\n--- 4. Generando Gráfica de Rendimiento ---")

# Aplanar los arrays para el gráfico
y_test_flat = y_test.flatten()
y_pred_flat = y_pred_test.flatten()

# Crear un rango ideal (la línea perfecta donde y_pred = y_test)
max_val = max(y_test_flat.max(), y_pred_flat.max())
min_val = min(y_test_flat.min(), y_pred_flat.min())
ideal_line = np.linspace(min_val, max_val, 10)


plt.figure(figsize=(7, 7))
# Puntos: Precio Real (Eje X) vs. Precio Predicto (Eje Y)
plt.scatter(y_test_flat, y_pred_flat, color='purple', label="Predicciones de Prueba")
# Línea ideal: El modelo perfecto caería en esta línea (y=x)
plt.plot(ideal_line, ideal_line, color='red', linestyle='--', label="Predicción Ideal (y=x)")

plt.xlabel("Precio Real (Y_test)")
plt.ylabel("Precio Predicto (Y_pred)")
plt.title("Rendimiento del Modelo: Real vs. Predicción")
plt.legend()
plt.grid(True)
plt.show()

# ==============================================================================
# 5. GUARDAR EL MODELO
# ==============================================================================
nombre_archivo = 'modelo_casa_4variables.joblib'
joblib.dump(model_4var, nombre_archivo)
print(f"\nModelo guardado como: {nombre_archivo}")