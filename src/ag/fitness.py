import numpy as np
from sklearn.metrics import r2_score

def fitness_poblacion(poblacion, atr_train, obj_train):
    fitness = np.zeros(poblacion.shape[0])

    for i in range(poblacion.shape[0]):
        fitness[i] = fitness(poblacion[i], atr_train, obj_train)

    return fitness

def fitness(individuo, atr_train, obj_train):
    nAtr = atr_train.shape[1]
    casos = atr_train.shape[0]
    y_pred = np.zeros(casos)

    # Regresión
    for i in range(casos):
        y_pred[i] = regresion(individuo, nAtr, atr_train[i])

    r2 = r2_score(obj_train, y_pred)

    penalizacion = 0

    if(r2 < 0):
        penalizacion = 100

    fitness = r2 - penalizacion

    return fitness

def regresion(coeficientes, nATr, caso):
    i = 0
    j = 0
    sol = 0

    while(i<nATr-1):
        sol += coeficientes[i]*caso[j]^coeficientes[i+1]
        i = i+2
        j = j+1

    sol += coeficientes[i]

    return sol