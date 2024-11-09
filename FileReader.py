import os
import numpy as np

class FileReader:
    """Clase para extraer los datos del archivo"""

    def __init__(self, fileName):
        self.fileName = fileName
        self.arrInf = np.array([])
        self.arregloDeClase = []
        self.countPerClass = []
        self.casillaDeClase = 0
        self.related = None

        self.obtainInfoTxt()
        self.obtainClasses()
        self.countNumberPerClass()

    def obtainInfoTxt(self):
        """Clase para leer el archivo y guardar los datos"""
        with open(self.fileName, "r", encoding='utf-8-sig') as archivo:
            # Leemos todas las líneas del archivo
            lines = archivo.readlines()
            # Convertimos las líneas en un arreglo NumPy
            self.arrInf = np.array([list(map(float, line.strip().split(','))) for line in lines])
        self.casillaDeClase = self.arrInf.shape[1] - 1

    def obtainClasses(self):
        """Clase para obtener un arreglo con todas las clases"""
        self.arregloDeClase = np.unique(self.arrInf[:, -1])

    def countNumberPerClass(self):
        """Cuenta el total de elementos por clase"""
        self.countPerClass = [np.sum(self.arrInf[:, -1] == clase) for clase in self.arregloDeClase]

    def chi_squared(self, column1, column2):
        observed = np.histogram2d(column1, column2, bins=3)[0]
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        total = observed.sum()
        expected = np.outer(row_totals, col_totals) / total
        expected[expected == 0] = 1e-10  # Evitar la división por cero
        chi_squared_value = np.sum((observed - expected)**2 / expected)
        return chi_squared_value
    
    def getRelated(self):
        # Encontrar la combinación de columnas con la chi-cuadrado más baja
        min_chi_squared = np.inf

        for i in range(self.arrInf.shape[1] - 1):
            for j in range(i + 1, self.arrInf.shape[1] - 1):
                chi_squared_value = self.chi_squared(self.arrInf[:, i], self.arrInf[:, j])
                if chi_squared_value < min_chi_squared:
                    min_chi_squared = chi_squared_value
                    self.related = (i, j)


# #pruebas de la funcion

# fileName = "EJEMPLOS/Dt1.txt"
# print("File name:")
# print(fileName)
# fr = FileReader(fileName)

# print("Datos extraidos:")
# print(fr.arrInf)

# print("\nArreglo de clases:")
# print(fr.arregloDeClase)

# print("\nElementos por clase:")
# print(fr.countPerClass)

# fr.getRelated()

# print("Elementos mas relacionados: ")
# print(fr.related)
# print("indice1")
# print(type(fr.related[1]))