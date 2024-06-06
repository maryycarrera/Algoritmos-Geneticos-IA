import pandas as pd
import numpy as np
from operadores import cruzar, mutar
from seleccion import seleccionarPadres, seleccionarSiguientePob
from fitness import fitness, regresion, fitness_poblacion

def AG(datos_train, datos_test, seed, nInd, maxIter):
    # Cargar datos de entrenamiento
    datos_train = pd.read_csv(datos_train)
    X_train = datos_train.iloc[:, :-1].values
    y_train = datos_train.iloc[:, -1].values

    # Cargar datos de test
    datos_test = pd.read_csv(datos_test)
    X_test = datos_test.iloc[:, :-1].values

    #Generar población inicial
    num_atributos = X_train.shape[1]
    poblacion_inicial = poblacion_inicial(num_atributos, nInd, seed)

    # Iterar
    pc = 0.7
    pm = 0.1
    poblacion_solucion = algoritmo_genetico(poblacion_inicial, nInd, maxIter, pc, pm, X_train, y_train)
    mejor_individuo, y_pred = mejor(poblacion_solucion, X_train, y_train, num_atributos)

    return mejor_individuo, y_pred

def poblacion_inicial(nAtributos, nInd, semilla):
    np.random.seed(semilla)
    poblacion = np.random.randint(2, size=(nInd, nAtributos*2+1))
    return poblacion

def algoritmo_genetico(pob, nInd, maxIter, pc, pm, atr_train, obj_train):
    evalu_pob = fitness_poblacion(pob, atr_train, obj_train) 
    i=0
    while(i<maxIter):
        padres = seleccionarPadres(evalu_pob, pob)
        hijos = cruzar(padres, pc)
        hijos = mutar(hijos, pm)
        evalu_hijos = fitness_poblacion(hijos, atr_train, obj_train) 
        pob = seleccionarSiguientePob(pob, hijos, evalu_pob, evalu_hijos)
        i = i+1

    return pob 

def mejor(poblacion, atr_train, obj_train):
    mejor = 0
    mejor_individuo = np.zeros(poblacion.shape[1])
    for i in range(poblacion.shape[0]):
        fit = fitness(poblacion[i], atr_train, obj_train)
        if(fit > mejor):
            mejor = fit
            mejor_individuo = poblacion[i]

    y_pred = regresion(mejor_individuo, atr_train.shape[1], atr_train)

    return mejor_individuo, y_pred