import numpy as np
from funciones_prueba import FuncionesPrueba
from pso import PSO

DIMENSIONES = 10
N_PARTICULAS = 30
ITERACIONES = 100

nombres_funciones = ["Esfera", "Ackley", "Griewank", "Rastrigin", "Rosenbrock"]

print("=" * 70)
print(f"PRUEBA DE PSO EN 5 FUNCIONES (Dim={DIMENSIONES}, Iter={ITERACIONES})")
print("=" * 70)

for nombre in nombres_funciones:
    info = FuncionesPrueba.obtener_info(nombre)
    if info is None:
        print(f"Error: No se encontró información para {nombre}")
        continue

    funcion = info["func"]
    limites = info["limites"]

    optimizador = PSO(funcion, DIMENSIONES, limites, n_particulas=N_PARTICULAS, max_iter=ITERACIONES)

    mejor_pos, mejor_fit, historial, tiempo, iter_mejor = optimizador.ejecutar()

    print(f"\nFunción: {nombre}")
    print(f" -> Rango de búsqueda: {limites}")
    print(f" -> Mejor fitness: {mejor_fit:.6e}")
    print(f" -> Iter. mejor: {iter_mejor} / {ITERACIONES}")
    print(f" -> Tiempo: {tiempo:.3f} s")

    if mejor_fit < 1e-5:
        print(" -> ¡CONVERGENCIA EXITOSA! (muy cerca del óptimo 0)")
    else:
        print(" -> Convergencia parcial / posible óptimo local")

print("\n" + "=" * 70)
print("Fin de la prueba de PSO.")
