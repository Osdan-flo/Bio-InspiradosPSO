import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from funciones_prueba import FuncionesPrueba
from pso import PSO
from algoritmo_genetico import AlgoritmoGenetico

def asegurar_directorio_output():
    """
    Busca la carpeta 'output' al mismo nivel que 'src'.
    Si no existe, la crea.
    """
    # Obtenemos la ruta de ESTE archivo (src/main_comparativo.py)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    
    # Subimos un nivel y entramos a 'output'
    ruta_output = os.path.join(ruta_actual, '..', 'output')
    
    if not os.path.exists(ruta_output):
        os.makedirs(ruta_output)
        print(f"Carpeta creada: {ruta_output}")
    
    return ruta_output

def ejecutar_comparacion():
    # --- 1. Configuración del Experimento ---
    DIMENSIONES_LISTA = [10, 30]         # Número de variables, 
    POBLACION = 50            # Tamaño del enjambre / población
    ITERACIONES = 100         # Duración de la búsqueda (Generaciones)
    REPETICIONES = 30          

    # Lista de funciones a probar
    nombres_funciones = ["Esfera", "Ackley", "Griewank", "Rastrigin", "Rosenbrock"]

    # Definir variantes de PSO
    configuraciones_pso = {
        "PSO_estandar": {
            "w": 0.7, 
            "c1": 1.49618, 
            "c2": 1.49618,
            "descripcion": "Configuración estándar balanceada"
        },
        "PSO_exploracion": {
            "w": 0.9,      #  Inercia alta = más exploración
            "c1": 0.5,     #  Menos atracción personal
            "c2": 2.5,     #  Más atracción social
            "descripcion": "Favorece exploración global"
        },
        "PSO_explotacion": {
            "w": 0.4,      #  Inercia baja = más explotación
            "c1": 2.5,     #  Más atracción personal
            "c2": 0.5,     #  Menos atracción social
            "descripcion": "Favorece refinamiento local"
        }
    }
    
    # Directorio donde se guardarán las imágenes
    dir_salida = asegurar_directorio_output()

    print(f"{'='*70}")
    print(f"COMPARACIÓN COMPLETA: PSO (3 variantes) vs AG")
    print(f"Dimensiones: {DIMENSIONES_LISTA}, Iter={ITERACIONES}, Reps={REPETICIONES}")
    print(f"{'='*70}")

    # Diccionario para guardar rankings globales
    rankings_globales = {
        dim: {var: [] for var in list(configuraciones_pso.keys()) + ["AG_base"]}
        for dim in DIMENSIONES_LISTA
    }

    ruta_csv = os.path.join(dir_salida, "resultados_pso_vs_ag.csv")
    with open(ruta_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "dimension","funcion", "algoritmo", "variante", "repeticion",
            "mejor_final", "tiempo_total", "iter_mejor"
        ])

        # CICLO EXTERNO: Dimensiones
        for DIM in DIMENSIONES_LISTA:
            print(f"\n{''*35}")
            print(f"DIMENSIÓN: {DIM}D")
            print(f"{''*35}")

            for nombre in nombres_funciones:
                print(f"\n Procesando función: {nombre} ({DIM}D)...")
            
                # Obtener configuración de la función
                info = FuncionesPrueba.obtener_info(nombre)
                funcion = info["func"]
                limites = info["limites"]

                todas_variantes = {}

                # ciclo sobre variantes de PSO
                for config_nombre, params in configuraciones_pso.items():
                    print(f" Variante: {config_nombre}")
                
                    resultados = []

                    for rep in range(REPETICIONES):
                        semilla_base = 12345 + rep
                    
                        # --- EJECUCIÓN PSO ---
                        pso = PSO(funcion, DIM, limites, n_particulas=POBLACION, max_iter=ITERACIONES, semilla=semilla_base,
                                w=params["w"], c1=params["c1"], c2=params["c2"])
                        _, mejor_fit_pso, curva_pso, tiempo_pso, iter_mejor_pso = pso.ejecutar()
                        resultados.append((mejor_fit_pso, tiempo_pso, iter_mejor_pso, curva_pso))
                        writer.writerow([DIM, nombre, "PSO", config_nombre, rep,
                                        mejor_fit_pso, tiempo_pso, iter_mejor_pso])
                        
                    # Estadísticas de esta variante
                    todas_variantes[config_nombre] = resultados
                    pso_mejores = [r[0] for r in resultados]
                    pso_tiempos = [r[1] for r in resultados]
                    print(f"      Media: {np.mean(pso_mejores):.3e} ± {np.std(pso_mejores):.1e}")
                    print(f"      Tiempo medio:  {np.mean(pso_tiempos):.3f}s ± {np.std(pso_tiempos):.3f}")
                        
                # Luego ejecuta AG (una sola vez, sin variantes)
                print(f"  Algoritmo Genético")
                resultados_ag = []
                
                for rep in range(REPETICIONES):
                    semilla_base = 12345 + rep + 10000
                    
                    ag = AlgoritmoGenetico(funcion, DIM, limites, 
                                        tam_poblacion=POBLACION, generaciones=ITERACIONES,
                                        prob_cruza=0.9, prob_mut=0.1, 
                                        semilla=semilla_base)
                    _, mejor_fit_ag, curva_ag, tiempo_ag, iter_mejor_ag = ag.ejecutar()
                    
                    resultados_ag.append((mejor_fit_ag, tiempo_ag, iter_mejor_ag, curva_ag))
                    writer.writerow([DIM, nombre, "AG", "AG_base", rep,  
                                mejor_fit_ag, tiempo_ag, iter_mejor_ag])
                    
                todas_variantes["AG_base"] = resultados_ag
                ag_mejores = [r[0] for r in resultados_ag]
                ag_tiempos = [r[1] for r in resultados_ag]
                print(f"      Media: {np.mean(ag_mejores):.3e} ± {np.std(ag_mejores):.1e}")
                print(f"      Tiempo medio:  {np.mean(ag_tiempos):.3f}s ± {np.std(pso_tiempos):.3f}")

                #  CALCULAR RANKING para esta función
                medias_variantes = {
                    var_nombre: np.mean([r[0] for r in res])
                    for var_nombre, res in todas_variantes.items()
                }
                # Ordenar de menor a mayor (mejor = menor fitness)
                ranking_orden = sorted(medias_variantes.items(), key=lambda x: x[1])

                print(f"\n    Ranking para {nombre} ({DIM}D):")
                for posicion, (var_nombre, media) in enumerate(ranking_orden, 1):
                    print(f"      {posicion}. {var_nombre}: {media:.3e}")
                    rankings_globales[DIM][var_nombre].append(posicion)

                plt.figure(figsize=(10, 6))
                datos_boxplot = []
                etiquetas = []
                    
                for var_nombre in list(configuraciones_pso.keys()) + ["AG_base"]:
                    datos_boxplot.append([r[0] for r in todas_variantes[var_nombre]])
                    etiquetas.append(var_nombre)

                plt.boxplot(datos_boxplot, labels=etiquetas)
                plt.title(f'{nombre} - {DIM}D: Distribución de fitness')
                plt.ylabel('Mejor fitness (log)')
                plt.yscale('log')
                plt.xticks(rotation=15, ha='right')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(dir_salida, f'boxplot_{nombre}_{DIM}D.png'))
                plt.close()

                #  Gráfica de convergencia (primera repetición de cada variante)
                plt.figure(figsize=(10, 6))
                for var_nombre in list(configuraciones_pso.keys()) + ["AG_base"]:
                    curva = todas_variantes[var_nombre][0][3]
                    estilo = '--' if var_nombre == "AG_base" else '-'
                    plt.plot(curva, label=var_nombre, linestyle=estilo, linewidth=2)
                    
                plt.yscale('log')
                plt.xlabel('Iteración')
                plt.ylabel('Mejor fitness (log)')
                plt.title(f'Convergencia: {nombre} - {DIM}D')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(dir_salida, f'convergencia_{nombre}_{DIM}D.png'))
                plt.close()

                

    #  TABLA DE RANKING FINAL
    print(f"\n{'='*70}")
    print(" RANKING PROMEDIO POR DIMENSIÓN")
    print(f"{'='*70}")
    
    for dim in DIMENSIONES_LISTA:
        print(f"\n DIMENSIÓN {dim}D:")
        print(f"{'Variante':<20} | Ranking Promedio")
        print("-" * 45)
        
        ranking_final = {
            var_nombre: np.mean(posiciones)
            for var_nombre, posiciones in rankings_globales[dim].items()
        }
        
        # Ordenar de mejor a peor
        for var_nombre, rank_prom in sorted(ranking_final.items(), key=lambda x: x[1]):
            print(f"{var_nombre:<20} | {rank_prom:.2f}")

    print(f"\n{'='*70}")
    print(f" EXPERIMENTOS COMPLETOS")
    print(f"Resultados: {ruta_csv}")
    print(f"Gráficas: {dir_salida}")
    print(f"{'='*70}")

if __name__ == "__main__":
    ejecutar_comparacion()