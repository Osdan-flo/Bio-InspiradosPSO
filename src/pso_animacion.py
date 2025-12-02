import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from funciones_prueba import FuncionesPrueba
from pso import PSO

# =========================
# Configuración básica
# =========================
NOMBRE_FUNCION = "Rosenbrock"   # "Esfera" o "Rastrigin" van muy bien para demo
DIMENSIONES = 2                # ¡2D para poder dibujarlo!
N_PARTICULAS = 30
ITERACIONES = 80               # pon algo moderado para que la animación no sea eterna
SEMILLA = 42

# =========================
# Preparar función y paisaje 2D
# =========================
info = FuncionesPrueba.obtener_info(NOMBRE_FUNCION)
funcion = info["func"]
lim_min, lim_max = info["limites"]

# Para que se vea bonito, acota un poco el rango si es muy grande
if NOMBRE_FUNCION == "Esfera":
    lim_min, lim_max = -5, 5
elif NOMBRE_FUNCION == "Rastrigin":
    lim_min, lim_max = -5.12, 5.12

def construir_paisaje(f, lim_min, lim_max, puntos=150):
    x = np.linspace(lim_min, lim_max, puntos)
    y = np.linspace(lim_min, lim_max, puntos)
    X, Y = np.meshgrid(x, y)
    
    # Vectorizamos el cálculo de f(x,y)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.array([f(p) for p in XY]).reshape(X.shape)
    return X, Y, Z

X, Y, Z = construir_paisaje(funcion, lim_min, lim_max)

# =========================
# Correr PSO y guardar trayectorias
# =========================
pso = PSO(funcion_objetivo=funcion,
          dimensiones=DIMENSIONES,
          limites=(lim_min, lim_max),
          n_particulas=N_PARTICULAS,
          max_iter=ITERACIONES,
          semilla=SEMILLA)

mejor_pos, mejor_fit, hist_conv, tiempo_total, iter_mejor, historial_pos = pso.ejecutar_con_historial()

print(f"Mejor fitness encontrado: {mejor_fit:.3e}")
print(f"Tiempo total: {tiempo_total:.3f} s")

# =========================
# Configurar figura para animación
# =========================
fig, ax = plt.subplots(figsize=(6, 5))

# Mapa de calor / curvas de nivel del paisaje
cont = ax.contourf(X, Y, Z, levels=40)
plt.colorbar(cont, ax=ax)

# Dispersión para las partículas
scatter = ax.scatter([], [], s=30, edgecolor="black")

# Punto para el mejor global actual
(best_point,) = ax.plot([], [], marker="*", markersize=12)

ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title(f"PSO en {NOMBRE_FUNCION} 2D")

# =========================
# Funciones de animación
# =========================
def init():
    scatter.set_offsets(np.empty((0, 2)))
    best_point.set_data([], [])
    return scatter, best_point

def update(frame):
    pos = historial_pos[frame]  # (n_part, 2)
    scatter.set_offsets(pos[:, :2])

    # Mejor partícula de ESA iteración
    fitness_iter = np.array([funcion(p) for p in pos])
    idx_best = np.argmin(fitness_iter)
    best = pos[idx_best]   # shape (2,)

    # 👉 OJO: pasar listas, no escalares
    best_point.set_data([best[0]], [best[1]])

    ax.set_title(
        f"PSO en {NOMBRE_FUNCION} 2D - Iter {frame+1}/{len(historial_pos)}"
    )
    return scatter, best_point

anim = FuncAnimation(
    fig,
    update,
    frames=len(historial_pos),
    init_func=init,
    blit=True,
    interval=100,   # ms entre frames (ajusta velocidad)
)

# Para verlo en vivo:
plt.tight_layout()

writer = PillowWriter(fps=15)
anim.save("pso_esfera_2d.gif", writer=writer)

plt.show()

# Si quieres guardarlo como video (necesitas ffmpeg instalado):
# anim.save("pso_{}_2D.mp4".format(NOMBRE_FUNCION.lower()), fps=15)
