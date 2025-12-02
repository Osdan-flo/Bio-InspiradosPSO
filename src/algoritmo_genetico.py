import copy
import time
import numpy as np

class IndividuoReal:
    def __init__(self, dimensiones, limites, rng):
        self.dimensiones = dimensiones
        self.limites = limites # Tupla (min, max)
        self.rng = rng
        self.fitness = float('inf')
        
        # Inicialización aleatoria: vector numpy para velocidad
        min_val, max_val = limites
        self.genotipo = self.rng.uniform(min_val, max_val, size=dimensiones)

    def evaluar(self, funcion_objetivo):
        # Evalúa el vector en la función (Esfera, Rastrigin, etc.)
        self.fitness = funcion_objetivo(self.genotipo)
        return self.fitness

class AlgoritmoGenetico:
    def __init__(self, funcion_objetivo, dimensiones, limites, tam_poblacion=30, generaciones=100, prob_cruza=0.9, prob_mut=0.1, semilla = None):
        self.func = funcion_objetivo
        self.dim = dimensiones
        self.limites = limites
        self.tam_pob = tam_poblacion
        self.generaciones = generaciones
        self.prob_cruza = prob_cruza
        self.prob_mut = prob_mut
        
        # Usamos numpy para semilla aleatoria también
        self.rng = np.random.default_rng(semilla)
        
        # Inicializar población
        self.poblacion = [IndividuoReal(self.dim, self.limites, self.rng) for _ in range(self.tam_pob)]
        
        # Evaluar inicial
        for ind in self.poblacion:
            ind.evaluar(self.func)
            
        # Mejor histórico
        self.mejor_global = min(self.poblacion, key=lambda x: x.fitness)
        self.historial_convergencia = [self.mejor_global.fitness]

    def _cruza_aritmetica(self, p1, p2):
        """Cruza: Promedio ponderado de los padres."""
        hijo1 = copy.deepcopy(p1)
        hijo2 = copy.deepcopy(p2)
        
        if self.rng.random() < self.prob_cruza:
            # Alpha aleatorio para dar variedad
            alpha = self.rng.random() 
            
            # Fórmula de cruza aritmética vectorizada
            hijo1.genotipo = alpha * p1.genotipo + (1 - alpha) * p2.genotipo
            hijo2.genotipo = (1 - alpha) * p1.genotipo + alpha * p2.genotipo
            
        return hijo1, hijo2

    def _mutacion_gaussiana(self, individuo, sigma=0.5):
        """Mutación: Suma ruido gaussiano a los genes."""
        # Máscara booleana para decidir qué genes mutan
        mask = self.rng.random(self.dim) < self.prob_mut
        
        if np.any(mask):
            ruido = self.rng.normal(0, sigma, size=self.dim)
            individuo.genotipo[mask] += ruido[mask]
            
            # Control de límites 
            min_l, max_l = self.limites
            individuo.genotipo = np.clip(individuo.genotipo, min_l, max_l)

    def ejecutar(self):
        """Ciclo principal del AG."""
        sigma_mutacion = 0.1 * (self.limites[1] - self.limites[0]) # 10% del rango como sigma inicial
        t0 = time.perf_counter()
        iter_mejor = 0

        for g in range(self.generaciones):
            nueva_poblacion = []
            
            # 1. Elitismo: El mejor pasa directo
            mejor_actual = min(self.poblacion, key=lambda x: x.fitness)
            nueva_poblacion.append(copy.deepcopy(mejor_actual))
            
            # 2. Generar resto de la población
            while len(nueva_poblacion) < self.tam_pob:
                # Selección por Torneo (tamaño 3)
                padres_candidatos = self.rng.choice(self.poblacion, size=6, replace=True)
                
                # Torneo 1
                torneo1 = padres_candidatos[0:3]
                p1 = min(torneo1, key=lambda x: x.fitness)
                
                # Torneo 2
                torneo2 = padres_candidatos[3:6]
                p2 = min(torneo2, key=lambda x: x.fitness)
                
                # Cruza
                h1, h2 = self._cruza_aritmetica(p1, p2)
                
                # Mutación
                self._mutacion_gaussiana(h1, sigma_mutacion)
                self._mutacion_gaussiana(h2, sigma_mutacion)
                
                # Evaluar
                h1.evaluar(self.func)
                h2.evaluar(self.func)
                
                nueva_poblacion.append(h1)
                if len(nueva_poblacion) < self.tam_pob:
                    nueva_poblacion.append(h2)
            
            self.poblacion = nueva_poblacion
            
            # Registrar estadísticas
            mejor_gen = min(self.poblacion, key=lambda x: x.fitness)
            
            # Actualizar mejor histórico si es necesario
            if mejor_gen.fitness < self.mejor_global.fitness:
                self.mejor_global = copy.deepcopy(mejor_gen)
                iter_mejor = g
                
            self.historial_convergencia.append(self.mejor_global.fitness)

        tiempo_total = time.perf_counter() - t0
            
        return (self.mejor_global.genotipo, self.mejor_global.fitness, self.historial_convergencia, tiempo_total, iter_mejor)