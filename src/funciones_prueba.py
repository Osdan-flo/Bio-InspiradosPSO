import numpy as np

class FuncionesPrueba:
    """
    Funciones benchmark para optimización continua.
    Todas las funciones aceptan un vector numpy 'x' y devuelven un escalar (aptitud/costo).
    """

    @staticmethod
    def esfera(x):
        """
        Función Esfera (Sphere).
        Unimodal, convexa. Mínimo global 0 en [0,0,...].
        """
        return np.sum(x**2)

    @staticmethod
    def ackley(x):
        """
        Función Ackley.
        Multimodal. Mínimo global 0 en [0,0,...].
        Tiene muchos picos pequeños y un gran 'embudo' central.
        """
        dim = len(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        
        suma1 = np.sum(x**2)
        suma2 = np.sum(np.cos(c * x))
        
        termino1 = -a * np.exp(-b * np.sqrt(suma1 / dim))
        termino2 = -np.exp(suma2 / dim)
        
        return termino1 + termino2 + a + np.exp(1)

    @staticmethod
    def griewank(x):
        """
        Función Griewank.
        Multimodal. Mínimo global 0 en [0,0,...].
        """
        # Indices base-1 para el producto (1, 2, 3...)
        indices = np.arange(1, len(x) + 1)
        
        parte_suma = np.sum(x**2) / 4000
        parte_prod = np.prod(np.cos(x / np.sqrt(indices)))
        
        return parte_suma - parte_prod + 1

    @staticmethod
    def rastrigin(x):
        """
        Función Rastrigin.
        Multimodal, compleja con muchos mínimos locales regularmente distribuidos.
        Mínimo global 0 en [0,0,...].
        """
        A = 10
        dim = len(x)
        return A * dim + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    @staticmethod
    def rosenbrock(x):
        """
        Función Rosenbrock.
        Unimodal/Multimodal según dimensión. Mínimo global 0 en [1,1,...].
        Muy plana, difícil convergencia final.
        """
        # Se calcula sumando 100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    @staticmethod
    def obtener_info(nombre_funcion):
        """
        Devuelve la función ejecutable y los límites de búsqueda recomendados.
        """
        info = {
            "Esfera":     {"func": FuncionesPrueba.esfera,     "limites": (-100, 100)},
            "Ackley":     {"func": FuncionesPrueba.ackley,     "limites": (-32, 32)},
            "Griewank":   {"func": FuncionesPrueba.griewank,   "limites": (-600, 600)},
            "Rastrigin":  {"func": FuncionesPrueba.rastrigin,  "limites": (-5.12, 5.12)},
            "Rosenbrock": {"func": FuncionesPrueba.rosenbrock, "limites": (-30, 30)}
        }
        return info.get(nombre_funcion)