"""Clase K-NN"""
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

class KNN:
    def __init__(self, dataSet, folds, k_neighbors, stratified):
        self.dataSet = dataSet
        self.foldsNum = folds
        self.k_neighbors = k_neighbors
        self.exactitudMedia = 0
        self.stratified = stratified
        self.noisyElements = set()
        self.correctClassifArr = []
        self.cleanDataList = []

        self.trainDataArray, self.testDataArray = self.dataValidation() # guarda el indice

    def setArregloDeClase(self, arregloClase):
        self.arregloDeClase = arregloClase
    
    def dataValidation(self):
        """Haciendo uso de una biblioteca se divide el dataset"""
        trainList = []
        testList = []
        
        caracteristics = [row[:-1] for row in self.dataSet]  # Caracteristicas
        classLabel = [row[-1] for row in self.dataSet]  # Clasificacion

        if self.stratified == True:
            skf = StratifiedKFold(n_splits=self.foldsNum, shuffle=True)  # Creamos el modelo de validacion estrificada
        else:
            skf = KFold(n_splits=self.foldsNum, shuffle=True)  # K-fold

        # Generar y guardar pliegues echos por el modelo
        for trainIndex, testIndex in skf.split(caracteristics, classLabel):
            trainList.append(trainIndex.tolist())
            testList.append(testIndex.tolist())

        trainData = trainList
        testData = testList

        return trainData, testData

    def getDistEuclid(self, pointA, pointB):
        suma = 0
        for atributo in range(len(pointA)):
            suma += (pointA[atributo] - pointB[atributo]) ** 2
        return math.sqrt(suma)
    
    def make_KNN(self):
        """Esta funcion llevara a cabo el algoritmo"""
        exactitud = 0
        # Ciclo para usar todos los folds en los que se dividió el dataset
        for i in range(self.foldsNum):
            observedClases = []
            correctClassif = 0
            cleanData = []
            
            # Obtenemos las distancias más cercanas de cada elemento
            for newExampleIndex in self.testDataArray[i]:
                distancesList = []
                
                for clasifiedIndex in self.trainDataArray[i]:
                    newDistance = self.getDistEuclid(self.dataSet[newExampleIndex][:-1], self.dataSet[clasifiedIndex][:-1])
                    distancesList.append((newDistance, self.dataSet[clasifiedIndex][-1]))

                distancesList.sort()  # Ordenamos arreglo de distancias de menor a mayor

                nearest = distancesList[:self.k_neighbors]  # Seleccionamos los más cercanos

                # Lista de clases de los k vecinos más cercanos
                nearest_classes = [clase for _, clase in nearest]
                
                # Obtenemos la clase más común entre los vecinos más cercanos
                max_class = max(nearest_classes, key=nearest_classes.count)

                observedClases.append(max_class)

                # Comprobamos si la clasificación es correcta
                if max_class == self.dataSet[newExampleIndex][-1]:
                    correctClassif += 1
                    cleanData.append(newExampleIndex)  # Agregamos elementos clasificados correctamente al conjunto de datos limpio
                else:
                    self.noisyElements.add(newExampleIndex)  # Guardamos el indice de los elementos con ruido

            self.cleanDataList.append(cleanData)
                
            # Datos clasificados correctamente en el K-fold
            self.correctClassifArr.append(correctClassif)

            # Calculamos la exactitud para este fold
            exactitud += (correctClassif / len(self.testDataArray[i])) * 100

        # Calculamos la exactitud media
        self.exactitudMedia = exactitud / self.foldsNum

    # def make_ENN(self):
    #     """Esta funcion llevara a cabo el algoritmo ENN"""
    #     cleanDataList = [[]]

    #     # Ciclo para usar todos los folds en los que se dividió el dataset
    #     for i in range(self.foldsNum):
    #         cleanData = []  # Inicializar conjunto de datos limpio para este pliegue

    #         # Obtenemos las distancias más cercanas de cada elemento
    #         for newExampleIndex in self.testDataArray[i]:
    #             distancesList = []
                
    #             for clasifiedIndex in self.trainDataArray[i]:
    #                 newDistance = self.getDistEuclid(self.dataSet[newExampleIndex][:-1], self.dataSet[clasifiedIndex][:-1])
    #                 distancesList.append((newDistance, self.dataSet[clasifiedIndex][-1]))

    #             distancesList.sort()  # Ordenamos arreglo de distancias de menor a mayor

    #             nearest = distancesList[:self.k_neighbors]  # Seleccionamos los más cercanos

    #             # Lista de clases de los k vecinos más cercanos
    #             nearest_classes = [clase for _, clase in nearest]
                
    #             # Obtenemos la clase más común entre los vecinos más cercanos
    #             max_class = max(nearest_classes, key=nearest_classes.count)

    #             # Comparamos la clase del objeto original con la que se le ha otorgado
    #             # y comprobamos si la clasificación es correcta
    #             if max_class == self.dataSet[newExampleIndex][-1]:
    #                 cleanData.append(self.dataSet[newExampleIndex])  # Agregamos elementos clasificados correctamente al conjunto de datos limpio
    #             else:
    #                 self.noisyElements.add(newExampleIndex)  # Convertimos el ndarray a una tupla antes de agregarlo al conjunto

    #         cleanDataList.append(cleanData)

    #     return cleanDataList
