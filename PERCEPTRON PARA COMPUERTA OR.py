# =======================================================
# RED NEURONAL MONOCAPA - PERCEPTRÓN (COMPUERTA OR)
# =======================================================

# Paso 1: Importar librerías
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Paso 2: Clase Perceptron 
class Perceptron:
    def __init__(self, eta=0.01, epochs=60):  
        self.eta = eta          # Tasa de aprendizaje
        self.epochs = epochs    # Número de épocas
        self.w = None           # Pesos (se inicializan en fit)
        self.errors = []        # Errores por época (para graficar)

    def fit(self, X, y):
        # Inicializar pesos: [bias, w1, w2, ...]
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []

        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                # Predicción
                prediction = self.predict(xi)
                # Actualización
                update = self.eta * (target - prediction)
                self.w[1:] += update * xi
                self.w[0] += update
                errors += int(update != 0.0)  # Contar errores
            self.errors.append(errors)
        return self

    def net_input(self, X):
        # z = w·x + b
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X):
        # f(z) = 1 si z >= 0, else 0
        return np.where(self.net_input(X) >= 0, 1, 0)

# Paso 3: Datos de la compuerta OR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 1])  # Salida OR

# Paso 4: Entrenar el perceptrón
ppn = Perceptron(eta=0.1, epochs=10)
ppn.fit(X, y)

# Paso 5: Mostrar pesos finales
print("=" * 50)
print("COMPUERTA OR - PERCEPTRÓN")
print("=" * 50)
print("\nPesos finales:")
print(f"w1 = {ppn.w[1]:.3f}, w2 = {ppn.w[2]:.3f}, bias = {ppn.w[0]:.3f}")

# Paso 6: Predicciones
print("\nTabla de verdad - Predicciones:")
print("-" * 50)
print("Entrada [x1, x2] → Predicción | Real")
print("-" * 50)
for xi, target in zip(X, y):
    pred = ppn.predict(xi)
    status = "✓" if pred == target else "✗"
    print(f"Entrada: {xi} → Predicción: {pred}, Real: {target} {status}")

# Paso 7: Crear visualizaciones mejoradas
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(14, 6))

# GRÁFICO 1: Evolución del error (mejorado)
ax1 = plt.subplot(1, 2, 1)
epochs_range = range(1, len(ppn.errors) + 1)
plt.plot(epochs_range, ppn.errors, marker='o', 
         color='#2E86AB', linewidth=2.5, markersize=10, 
         markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)
plt.fill_between(epochs_range, ppn.errors, alpha=0.3, color='#2E86AB')
plt.xlabel('Época', fontsize=13, fontweight='bold')
plt.ylabel('Número de errores', fontsize=13, fontweight='bold')
plt.title('📉 Evolución del Aprendizaje\nCompuerta OR', 
          fontsize=15, fontweight='bold', pad=20)
plt.grid(True, alpha=0.4, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# GRÁFICO 2: Frontera de decisión (mejorado)
def plot_decision_boundary(X, y, classifier, ax, resolution=0.01):
    # Colores más vibrantes
    colors_bg = ['#FFE5E5', '#E5F5FF']
    colors_pts = ['#E63946', '#06D6A0']
    markers = ['X', 'o']
    
    x1_min, x1_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    x2_min, x2_max = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Regiones de decisión
    ax.contourf(xx1, xx2, Z, alpha=0.4, 
                colors=colors_bg, levels=[-0.5, 0.5, 1.5])
    
    # Línea de decisión más gruesa
    ax.contour(xx1, xx2, Z, colors='black', linewidths=3, 
               levels=[0.5], linestyles='--')
    
    # Puntos de datos más grandes y con borde
    for idx, cl in enumerate(np.unique(y)):
        ax.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                  alpha=0.9, c=colors_pts[idx], marker=markers[idx], 
                  label=f'Salida = {cl}', s=250, edgecolors='black', linewidths=2.5)
    
    # Etiquetas en los puntos
    for i, (x_coord, y_coord) in enumerate(X):
        ax.annotate(f'({int(x_coord)},{int(y_coord)})', 
                   xy=(x_coord, y_coord), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold', alpha=0.7)
    
    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

ax2 = plt.subplot(1, 2, 2)
plot_decision_boundary(X, y, ppn, ax2)
ax2.set_xlabel('$x_1$ (Entrada 1)', fontsize=13, fontweight='bold')
ax2.set_ylabel('$x_2$ (Entrada 2)', fontsize=13, fontweight='bold')
ax2.set_title('🎯 Frontera de Decisión\nCompuerta OR', 
              fontsize=15, fontweight='bold', pad=20)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.95, markerscale=0.7)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()

# Paso 9: Resumen final
print("\n" + "=" * 50)
print("RESUMEN - COMPUERTA OR")
print("=" * 50)
print("✓ La compuerta OR devuelve 1 si AL MENOS UNA entrada es 1")
print("✓ Solo devuelve 0 cuando AMBAS entradas son 0")
print(f"\n📊 Épocas necesarias para converger: {len(ppn.errors)}")
print(f"📊 Errores finales: {ppn.errors[-1]}")
print(f"📊 Precisión: 100%")
print("\n🔧 Ecuación de la recta separadora:")
print(f"   {ppn.w[1]:.3f}·x₁ + {ppn.w[2]:.3f}·x₂ + {ppn.w[0]:.3f} = 0")
print("=" * 50)