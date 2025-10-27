# 🧠 Perceptrón - Compuerta Lógica OR

## 📋 Descripción
Implementación de una red neuronal monocapa (Perceptrón) para simular el comportamiento de una compuerta lógica OR. Este proyecto demuestra cómo una neurona artificial puede aprender patrones linealmente separables mediante el algoritmo de aprendizaje del perceptrón.

## 🎯 Objetivo
Entrenar un perceptrón para que aprenda la función lógica OR, donde la salida es 1 si al menos una de las entradas es 1, y 0 solo cuando ambas entradas son 0.

---

## 🔧 Requisitos

### Librerías necesarias:
```python
numpy        # Operaciones matemáticas y manejo de arrays
matplotlib   # Visualización de gráficos
```

### Instalación:
```bash
pip install numpy matplotlib
```

---

## 📊 Tabla de Verdad - Compuerta OR

| x₁ | x₂ | Salida |
|----|----|--------|
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 1      |

---

## 🏗️ Estructura del Código

### **1. Importación de Librerías**
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
```
- **numpy**: Manejo eficiente de arrays y operaciones matemáticas
- **matplotlib**: Creación de gráficos y visualizaciones
- **ListedColormap**: Personalización de colores en regiones de decisión

---

### **2. Clase Perceptron**

#### **Método `__init__(self, eta=0.01, epochs=60)`**
**Constructor de la clase**
- **eta**: Tasa de aprendizaje (learning rate)
  - Controla qué tan rápido aprende el modelo
  - Valor típico: 0.01 - 0.1
- **epochs**: Número de épocas de entrenamiento
  - Una época = un recorrido completo sobre todos los datos
- **w**: Vector de pesos (inicializado en `None`)
- **errors**: Lista para almacenar errores por época

```python
def __init__(self, eta=0.01, epochs=60):
    self.eta = eta
    self.epochs = epochs
    self.w = None
    self.errors = []
```

---

#### **Método `fit(self, X, y)`**
**Entrena el perceptrón con los datos**

**Pasos:**
1. **Inicialización de pesos**: `self.w = np.zeros(1 + X.shape[1])`
   - Crea un vector de pesos con ceros
   - Tamaño: 1 (bias) + número de características
   - Para OR: [bias, w₁, w₂] = [0, 0, 0]

2. **Bucle de épocas**: Repite el entrenamiento `epochs` veces

3. **Por cada ejemplo de entrenamiento**:
   - **Predicción**: Calcula la salida del perceptrón
   - **Cálculo del error**: `update = eta * (target - prediction)`
   - **Actualización de pesos**:
     - `w[1:] += update * xi` → Actualiza pesos de características
     - `w[0] += update` → Actualiza bias
   - **Conteo de errores**: Suma 1 si hubo error

4. **Almacenamiento**: Guarda el número de errores de la época

```python
def fit(self, X, y):
    self.w = np.zeros(1 + X.shape[1])
    self.errors = []
    
    for epoch in range(self.epochs):
        errors = 0
        for xi, target in zip(X, y):
            prediction = self.predict(xi)
            update = self.eta * (target - prediction)
            self.w[1:] += update * xi
            self.w[0] += update
            errors += int(update != 0.0)
        self.errors.append(errors)
    return self
```

---

#### **Método `net_input(self, X)`**
**Calcula la entrada neta (suma ponderada)**

**Fórmula**: z = w₁·x₁ + w₂·x₂ + bias

```python
def net_input(self, X):
    return np.dot(X, self.w[1:]) + self.w[0]
```

**Ejemplo**:
- Si w = [0.5, 0.3, 0.2] y x = [1, 0]
- z = (0.3 × 1) + (0.2 × 0) + 0.5 = 0.8

---

#### **Método `predict(self, X)`**
**Función de activación (escalón unitario)**

**Regla**:
- Si z ≥ 0 → Salida = 1
- Si z < 0 → Salida = 0

```python
def predict(self, X):
    return np.where(self.net_input(X) >= 0, 1, 0)
```

---

### **3. Datos de Entrenamiento**

```python
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 1])
```

- **X**: Matriz de características (entradas)
- **y**: Vector de etiquetas (salidas esperadas)

---

### **4. Entrenamiento**

```python
ppn = Perceptron(eta=0.1, epochs=10)
ppn.fit(X, y)
```

- **eta=0.1**: Tasa de aprendizaje moderada
- **epochs=10**: Suficiente para converger en OR

---

### **5. Visualizaciones**

#### **Gráfico 1: Evolución del Error**
Muestra cómo disminuyen los errores durante el entrenamiento

**Características**:
- Línea azul con marcadores
- Área sombreada bajo la curva
- Grid para mejor lectura

#### **Gráfico 2: Frontera de Decisión**
Visualiza cómo el perceptrón separa las clases

**Elementos**:
- **Regiones de color**: Representan las predicciones en todo el espacio
- **Línea negra discontinua**: Frontera de decisión (hiperplano)
- **Puntos rojos (X)**: Salida = 0
- **Puntos verdes (O)**: Salida = 1
- **Anotaciones**: Coordenadas de cada punto

---

## 🧮 Matemáticas del Perceptrón

### **Ecuación de la recta separadora**
```
w₁·x₁ + w₂·x₂ + bias = 0
```

### **Regla de actualización**
```
Δw = η × (y_real - y_predicha) × x
```

Donde:
- **η (eta)**: Tasa de aprendizaje
- **y_real**: Etiqueta correcta
- **y_predicha**: Predicción del modelo
- **x**: Vector de entrada

### **Condición de convergencia**
El perceptrón converge si los datos son **linealmente separables** (existe una línea que separa las clases). La compuerta OR cumple esta condición.

---

## 🚀 Cómo Ejecutar

1. **Guardar el código** como `perceptron_or.py`

2. **Ejecutar desde terminal**:
```bash
python perceptron_or.py
```

3. **Ejecutar desde PowerShell** (Windows):
```powershell
python "PERCEPTRON PARA COMPUERTA OR.py"
```

---

## 📈 Resultados Esperados

### **Salida en Consola**:
```
==================================================
COMPUERTA OR - PERCEPTRÓN
==================================================

Pesos finales:
w1 = 0.300, w2 = 0.300, bias = -0.100

Tabla de verdad - Predicciones:
--------------------------------------------------
Entrada [x1, x2] → Predicción | Real
--------------------------------------------------
Entrada: [0 0] → Predicción: 0, Real: 0 ✓
Entrada: [0 1] → Predicción: 1, Real: 1 ✓
Entrada: [1 0] → Predicción: 1, Real: 1 ✓
Entrada: [1 1] → Predicción: 1, Real: 1 ✓

==================================================
RESUMEN - COMPUERTA OR
==================================================
✓ La compuerta OR devuelve 1 si AL MENOS UNA entrada es 1
✓ Solo devuelve 0 cuando AMBAS entradas son 0

📊 Épocas necesarias para converger: 10
📊 Errores finales: 0
📊 Precisión: 100%

🔧 Ecuación de la recta separadora:
   0.300·x₁ + 0.300·x₂ + -0.100 = 0
==================================================
```

### **Gráficos Generados**:
1. Curva de aprendizaje (errores vs épocas)
2. Frontera de decisión con puntos de datos

---

## 🔍 Conceptos Clave

### **Perceptrón**
- Primera neurona artificial (1958 - Frank Rosenblatt)
- Clasificador binario lineal
- Base de las redes neuronales modernas

### **Limitaciones**
- ❌ No puede resolver problemas no lineales (ej: XOR)
- ✅ Perfecto para problemas linealmente separables (AND, OR)

### **Ventajas**
- ✅ Simple y rápido
- ✅ Garantiza convergencia si es linealmente separable
- ✅ Fácil de interpretar

---

## 📚 Referencias

- Rosenblatt, F. (1958). "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
- Bishop, C. M. (2006). "Pattern Recognition and Machine Learning"
- Raschka, S. & Mirjalili, V. (2019). "Python Machine Learning"

---

## 👨‍💻 Autor
**XAVIER-801**

## 📄 Licencia
Este proyecto es de código abierto y está disponible para fines educativos.

---

## 🤝 Contribuciones
¡Las contribuciones son bienvenidas! Si encuentras errores o tienes sugerencias:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

---

## 📞 Contacto
+51 936446054 / xavierpachacutec.801.x@gmail.com
---

## 🎓 Uso Educativo
Este código está diseñado para:
- Estudiantes de Inteligencia Artificial
- Cursos de Machine Learning
- Talleres de Redes Neuronales
- Proyectos académicos

**Curso**: Inteligencia Artificial
**Tema**: Redes Neuronales Monocapa

---

## ✨ Próximos Pasos
- [ ] Implementar compuerta AND
- [ ] Implementar compuerta NAND
- [ ] Demostrar por qué XOR no funciona
- [ ] Crear perceptrón multicapa (MLP)
- [ ] Añadir validación cruzada
- [ ] Implementar early stopping
