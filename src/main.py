import numpy as np
from funciones_prueba import FuncionesPrueba
from pso import PSO

# --- Configuración ---
DIMENSIONES = 10         # Dimensión del problema (número de variables)
N_PARTICULAS = 30        # Tamaño del enjambre
ITERACIONES = 100        # Duración de la búsqueda

# Lista de funciones a probar
nombres_funciones = ["Esfera", "Ackley", "Griewank", "Rastrigin", "Rosenbrock"]

print(f"{'='*70}")
print(f"PRUEBA DE PSO EN 5 FUNCIONES (Dim={DIMENSIONES}, Iter={ITERACIONES})")
print(f"{'='*70}")

for nombre in nombres_funciones:
    # 1. Obtener configuración de la función
    info = FuncionesPrueba.obtener_info(nombre)
    if info is None:
        print(f"Error: No se encontró información para {nombre}")
        continue
        
    funcion = info["func"]
    limites = info["limites"]
    
    # 2. Instanciar PSO
    optimizador = PSO(funcion, DIMENSIONES, limites, N_PARTICULAS, ITERACIONES)
    
    # 3. Ejecutar
    mejor_pos, mejor_fit, historial = optimizador.ejecutar()
    
    # 4. Mostrar resultados
    print(f"\nFunción: {nombre}")
    print(f" -> Rango de Búsqueda: {limites}")
    print(f" -> Mejor Fitness Encontrado: {mejor_fit:.6e}") # Notación científica
    
    # Interpretación simple del resultado
    if mejor_fit < 1e-5:
        print(" -> ¡CONVERGENCIA EXITOSA! (Llegó muy cerca del óptimo 0)")
    else:
        print(" -> Convergencia parcial (o atrapado en óptimo local)")

print(f"\n{'='*70}")
print("Fin de la ejecución de prueba.")