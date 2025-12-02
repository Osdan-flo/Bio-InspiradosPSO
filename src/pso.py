import numpy as np
import time

class PSO:
    def __init__(self, funcion_objetivo, dimensiones, limites, n_particulas=30, max_iter=100, semilla=None,
                 w= 0.7, c1=1.49618, c2=1.49618):
        """
        Inicializa el algoritmo de Optimización por Cúmulo de Partículas.
        
        Args:
            funcion_objetivo: La función matemática a minimizar.
            dimensiones: Número de variables (D).
            limites: Tupla (min, max) del espacio de búsqueda.
            n_particulas: Tamaño del enjambre.
            max_iter: Número máximo de iteraciones.
        """
        self.func = funcion_objetivo
        self.dim = dimensiones
        self.min_val, self.max_val = limites
        self.n_part = n_particulas
        self.max_iter = max_iter
        
        
        # --- Hiperparámetros Estándar ---
        self.w = w       # Inercia (peso de la velocidad anterior)
        self.c1 = c1  # Coeficiente cognitivo (atracción a su mejor personal)
        self.c2 = c2  # Coeficiente social (atracción al mejor del enjambre)
        
        # RNG propio (reproducible)
        self.rng = np.random.default_rng(semilla)

        # --- Inicialización del Enjambre ---
        # Posiciones aleatorias: Matriz de (n_particulas x dimensiones)
        self.posiciones = self.rng.uniform(self.min_val, self.max_val, size = (self.n_part, self.dim))
        
        # Velocidades aleatorias (pequeñas al inicio)
        self.velocidades = self.rng.uniform(-1, 1, size = (self.n_part, self.dim))
        
        # Mejor posición personal (pbest) inicial es la actual
        self.mejor_personal_pos = np.copy(self.posiciones)
        self.mejor_personal_fit = np.array([self.func(p) for p in self.posiciones])
        
        # Mejor global (gbest) del enjambre
        idx_mejor_global = np.argmin(self.mejor_personal_fit)
        self.mejor_global_pos = np.copy(self.mejor_personal_pos[idx_mejor_global])
        self.mejor_global_fit = self.mejor_personal_fit[idx_mejor_global]
        
        # Historial para gráficas (guardamos el mejor valor de cada iteración)
        self.historial_convergencia = []

    def ejecutar(self):
        """Ejecuta el ciclo principal del PSO."""

        t0 = time.perf_counter()
        iter_mejor = 0
        
        for t in range(self.max_iter):
            # 1. Actualizar Velocidades
            # r1, r2 son vectores aleatorios [0, 1] 
            r1 = self.rng.random((self.n_part, self.dim))
            r2 = self.rng.random((self.n_part, self.dim))
            
            # Ecuación fundamental de PSO (vectorizada):
            componente_inercia = self.w * self.velocidades
            componente_cognitivo = self.c1 * r1 * (self.mejor_personal_pos - self.posiciones)
            componente_social = self.c2 * r2 * (self.mejor_global_pos - self.posiciones)
            
            self.velocidades = componente_inercia + componente_cognitivo + componente_social

            # 2. Actualizar Posiciones
            # x(t+1) = x(t) + v(t+1)
            self.posiciones = self.posiciones + self.velocidades
            
            # 3. Control de Límites (Clamp)
            # Si se salen del mapa, los regresamos al borde
            self.posiciones = np.clip(self.posiciones, self.min_val, self.max_val)
            
            # 4. Evaluación y Actualización de Mejores (Bests)
            fitness_actuales = np.array([self.func(p) for p in self.posiciones])
            
            # Actualizar Mejor Personal (pbest)
            # Buscamos dónde el nuevo fitness es mejor que el histórico
            mejoras = fitness_actuales < self.mejor_personal_fit
            self.mejor_personal_pos[mejoras] = self.posiciones[mejoras]
            self.mejor_personal_fit[mejoras] = fitness_actuales[mejoras]
            
            # Actualizar Mejor Global (gbest)
            min_fitness_iter = np.min(fitness_actuales)
            idx_mejor_iter = np.argmin(fitness_actuales)
            
            if min_fitness_iter < self.mejor_global_fit:
                self.mejor_global_fit = min_fitness_iter
                self.mejor_global_pos = np.copy(self.posiciones[idx_mejor_iter])
                iter_mejor = t
            
            # Guardar dato para la gráfica
            self.historial_convergencia.append(self.mejor_global_fit)
        
        tiempo_total = time.perf_counter() - t0
            
        return (self.mejor_global_pos, self.mejor_global_fit, self.historial_convergencia, tiempo_total, iter_mejor)
    
    def ejecutar_con_historial(self):
        """
        Igual que ejecutar(), pero además guarda las posiciones del enjambre
        en cada iteración para poder animarlas.
        """
        import time
        t0 = time.perf_counter()
        iter_mejor = 0
        
        historial_posiciones = []   # lista de arrays (n_part x dim)
        self.historial_convergencia = []  # por si quieres reusar

        for t in range(self.max_iter):
            # 1. Actualizar Velocidades
            r1 = self.rng.random((self.n_part, self.dim))
            r2 = self.rng.random((self.n_part, self.dim))
            
            componente_inercia = self.w * self.velocidades
            componente_cognitivo = self.c1 * r1 * (self.mejor_personal_pos - self.posiciones)
            componente_social = self.c2 * r2 * (self.mejor_global_pos - self.posiciones)
            
            self.velocidades = componente_inercia + componente_cognitivo + componente_social

            # 2. Actualizar Posiciones
            self.posiciones = self.posiciones + self.velocidades
            self.posiciones = np.clip(self.posiciones, self.min_val, self.max_val)

            # 3. Evaluar y actualizar mejores
            fitness_actuales = np.array([self.func(p) for p in self.posiciones])

            mejoras = fitness_actuales < self.mejor_personal_fit
            self.mejor_personal_pos[mejoras] = self.posiciones[mejoras]
            self.mejor_personal_fit[mejoras] = fitness_actuales[mejoras]

            min_fitness_iter = np.min(fitness_actuales)
            idx_mejor_iter = np.argmin(fitness_actuales)

            if min_fitness_iter < self.mejor_global_fit:
                self.mejor_global_fit = min_fitness_iter
                self.mejor_global_pos = np.copy(self.posiciones[idx_mejor_iter])
                iter_mejor = t

            # guardar para gráficas
            self.historial_convergencia.append(self.mejor_global_fit)
            historial_posiciones.append(self.posiciones.copy())

        tiempo_total = time.perf_counter() - t0

        return (self.mejor_global_pos,
                self.mejor_global_fit,
                self.historial_convergencia,
                tiempo_total,
                iter_mejor,
                historial_posiciones)