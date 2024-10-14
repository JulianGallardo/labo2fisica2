import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parámetros de la malla
nx, ny = 101, 161  # Dimensiones de la malla (mínimo 101x161 puntos)
tolerancia = 1e-5  # Tolerancia para la convergencia
max_iter = 5000  # Número máximo de iteraciones

# Inicialización de la matriz del potencial (0 en todas partes al inicio)
potencial = np.zeros((nx, ny))

# Almacenar posiciones de los bornes para aplicar condiciones de Dirichlet
bornes = []

# Función para agregar bornes con condiciones de Dirichlet
def agregar_bornes():
    cantidad_bornes = int(input("¿Cuántos bornes deseas agregar? (2 o 3) "))
    for _ in range(cantidad_bornes):
        x_inicio = int(input(f"Ingrese la coordenada x de inicio del borne (0 a {nx - 1}): "))
        x_fin = int(input(f"Ingrese la coordenada x de fin del borne (0 a {nx - 1}): "))
        y_inicio = int(input(f"Ingrese la coordenada y de inicio del borne (0 a {ny - 1}): "))
        y_fin = int(input(f"Ingrese la coordenada y de fin del borne (0 a {ny - 1}): "))
        potencial_borne = float(input("Ingrese el potencial del borne (en voltios): "))

        # Establecer el potencial en el rango dado y almacenar el borne
        potencial[x_inicio:x_fin, y_inicio:y_fin] = potencial_borne
        bornes.append(((x_inicio, x_fin), (y_inicio, y_fin), potencial_borne))

# Función para imponer condiciones de Dirichlet en los bornes
def condiciones_dirichlet_bornes(potencial):
    for borne in bornes:
        (x_inicio, x_fin), (y_inicio, y_fin), potencial_borne = borne
        potencial[x_inicio:x_fin, y_inicio:y_fin] = potencial_borne

# Función para aplicar las condiciones de Neumann en los bordes de la malla
def condiciones_neumann(potencial):
    # Borde izquierdo: Φ(0, j) = Φ(1, j)
    potencial[0, :] = potencial[1, :]

    # Borde derecho: Φ(nx-1, j) = Φ(nx-2, j)
    potencial[-1, :] = potencial[-2, :]

    # Borde superior: Φ(i, 0) = Φ(i, 1)
    potencial[:, 0] = potencial[:, 1]

    # Borde inferior: Φ(i, ny-1) = Φ(i, ny-2)
    potencial[:, -1] = potencial[:, -2]

# Función para aplicar el método de relajación con las condiciones de Neumann
def relajacion(potencial, max_iter, tolerancia):
    for it in range(max_iter):
        potencial_nuevo = potencial.copy()
        # Actualización del potencial usando el promedio de los vecinos
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                if not any(
                        x_inicio <= i < x_fin and y_inicio <= j < y_fin for (x_inicio, x_fin), (y_inicio, y_fin), _ in
                        bornes):
                    potencial_nuevo[i, j] = 0.25 * (potencial[i + 1, j] + potencial[i - 1, j] +
                                                    potencial[i, j + 1] + potencial[i, j - 1])

        # Imponer condiciones de Neumann en todos los bordes
        condiciones_neumann(potencial_nuevo)

        # Reaplicar condiciones de Dirichlet a los bornes
        condiciones_dirichlet_bornes(potencial_nuevo)

        # Criterio de convergencia
        if np.max(np.abs(potencial_nuevo - potencial)) < tolerancia:
            print(f'Convergencia alcanzada en la iteración {it + 1}')
            break

        potencial = potencial_nuevo

    return potencial

# Llamada a la función para agregar bornes (Dirichlet)
agregar_bornes()

# Ejecutar el método de relajación
potencial_final = relajacion(potencial, max_iter, tolerancia)

# Graficar las líneas equipotenciales (gráfico de contorno)
plt.figure()
contour = plt.contourf(potencial_final.T, 50, cmap='viridis')  # Colormap más claro
plt.colorbar(contour, label='Potencial (V)')

# Añadir las líneas equipotenciales
plt.contour(potencial_final.T, levels=10, colors='white', linewidths=0.5)  # Líneas equipotenciales en blanco

plt.title('Líneas Equipotenciales')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal')
plt.show()

# Graficar la superficie 3D del potencial
x = np.linspace(0, nx - 1, nx)
y = np.linspace(0, ny - 1, ny)
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, potencial_final.T, cmap='viridis')  # Colormap más claro
ax.set_title('Superficie 3D del Potencial')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Potencial (V)')
plt.show()
