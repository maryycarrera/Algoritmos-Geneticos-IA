import pandas as pd
import numpy as np
from operadores import cruzar, mutar
from seleccion import seleccionarPadres, seleccionarSiguientePob
from fitness import regresion, fitness_poblacion

class AG:
    def __init__(self, datos_train, datos_test, seed, nInd, maxIter):
        self.datos_train = pd.read_csv(datos_train)
        self.X_train = self.datos_train.iloc[:, :-1].values
        self.y_train = self.datos_train.iloc[:, -1].values
        
        self.datos_test = pd.read_csv(datos_test)
        self.X_test = self.datos_test.iloc[:, :-1].values

        self.seed = seed
        self.nInd = nInd
        self.maxIter = maxIter

        self.pob_inicial = self.poblacion_inicial(self.X_train.shape[1], nInd, seed)
    
    def run(self):
        pc = 0.8
        pm = 0.1
        poblacion_solucion, evalu_pob = self.algoritmo_genetico(self.pob_inicial, self.nInd, self.maxIter, pc, pm, self.X_train, self.y_train)
        mejor_individuo, y_pred = self.mejor(poblacion_solucion,evalu_pob,self.X_test)
        return mejor_individuo, y_pred

    def poblacion_inicial(self, nAtributos, nInd, semilla):
        np.random.seed(semilla)
        poblacion = np.random.randint(low=-2, high=2, size=(nInd, nAtributos*2+1))
        return poblacion

    def algoritmo_genetico(self, pob, nInd, maxIter, pc, pm, atr_train, obj_train):
        evalu_pob = fitness_poblacion(pob, atr_train, obj_train)
        porcentaje_elitismo=0.1
        i = 0
        while(i < maxIter):
            padres = seleccionarPadres(evalu_pob, pob, porcentaje_elitismo, nInd)
            hijos = cruzar(padres, pc)
            hijos = mutar(hijos, pm)
            pob = seleccionarSiguientePob(pob, hijos, evalu_pob, porcentaje_elitismo)
            evalu_pob = fitness_poblacion(pob, atr_train, obj_train)
            i += 1

        return pob, evalu_pob

    def mejor(self, poblacion, evalu_pob, atr_test):
        mejor = evalu_pob[0]
        mejor_individuo = poblacion[0]
        for i in range(1,poblacion.shape[0]):
            fit = evalu_pob[i]  # Fitness con el conjunto de entrenamiento
            if(fit > mejor):
                mejor = fit
                mejor_individuo = poblacion[i]
            # if(fit < mejor):
            #     mejor = fit
            #     mejor_individuo = poblacion[i]

        y_pred = []
        for caso in atr_test:
            y_pred.append(regresion(mejor_individuo, atr_test.shape[1], caso))  # Predicciones para el conjunto de prueba

        return mejor_individuo, np.array(y_pred)