import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import random

def AG(datos_train, datos_test, seed, nInd, maxIter):
    # Cargar datos de entrenamiento
    datos_train = pd.read_csv(datos_train)
    X_train = datos_train.iloc[:, :-1].values
    y_train = datos_train.iloc[:, -1].values

    # Cargar datos de test
    datos_test = pd.read_csv(datos_test)
    X_test = datos_test.iloc[:, :-1].values

    #Generar población inicial
    nAtr = X_train.shape[1]
    poblacion_inicial = poblacion_inicial(nAtr, nInd, seed)

    # Iterar
    pc = 0.7
    pm = 0.1
    poblacion_solucion = algoritmo_genetico(poblacion_inicial, nInd, maxIter, pc, pm, nAtr, X_train, y_train)
    mejor_individuo, y_pred = mejor(poblacion_solucion)

    return mejor_individuo, y_pred

def poblacion_inicial(nAtributos, nInd, semilla):
    np.random.seed(semilla)
    poblacion = np.random.randint(2, size=(nInd, nAtributos*2+1))
    return poblacion

def algoritmo_genetico(pob, nInd, maxIter, pc, pm, nAtr, atr_train, obj_train):
    evalu_pob = fitness_poblacion(pob, atr_train, obj_train, nAtr) 
    i=0
    while(i<maxIter):
        padres = seleccionarPadres(evalu_pob,pob)
        hijos = cruzar(padres, pc)
        hijos = mutar(hijos, pm)
        evalu_hijos = fitness_poblacion(hijos, atr_train, obj_train, nAtr) 
        pob = seleccionarSiguientePob(pob, hijos, evalu_pob, evalu_hijos)
        i = i+1

    return pob 

def fitness_poblacion(poblacion, atr_train, obj_train, nAtr):
    fitness = np.zeros(poblacion.shape[0])

    for i in range(poblacion.shape[0]):
        fitness[i] = fitness(poblacion[i], atr_train, obj_train, nAtr)

    return fitness

def fitness(individuo, atr_train, obj_train, nAtr):
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

def seleccionarPadres(evalu_pob,pob):
    porcentaje=0.2
    evalu_pob_ordenado = sorted(evalu_pob, reverse=True)
    valores_maximos = evalu_pob_ordenado[:porcentaje_mejores_ind]

    # Emparejar fitness con individuos y ordenarlos por fitness en orden descendente
    evalu_pob_con_indices = list(enumerate(evalu_pob))
    evalu_pob_con_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Calcular el número de individuos a mantener como elitismo
    total_ind = len(evalu_pob)
    porcentaje_mejores_ind = int(total_ind * 0.2)
    
    # Obtener los índices de los mejores individuos
    indices_elitismo = [idx for idx, _ in evalu_pob_con_indices[:cantidad_elitismo]]
    
    # Obtener los individuos que se mantendrán como elitismo
    mejores_individuos = [pob[idx] for idx in indices_elitismo]
    
    # Obtener los individuos que serán cruzados (excluyendo los de elitismo)
    indices_cruce = [idx for idx, _ in evalu_pob_con_indices[cantidad_elitismo:]]
    individuos_cruce = [pob[idx] for idx in indices_cruce]
    
    return mejores_individuos, individuos_cruce

def cruzar(padres, pc):
    punto_cruce = padres.shape[1] / 2
    hijos = np.zeros(padres.shape)
    i = 0

    while(i<padres.shape[0]):
        padre1 = padres[i]
        padre2 = padres[i+1]
        rand = random.random()

        if(rand < pc):
            hijo1 = np.concatenate((padre1[:punto_cruce], padre2[punto_cruce:]))
            hijo2 = np.concatenate((padre2[:punto_cruce], padre1[punto_cruce:]))
        else:
            hijo1 = padre1
            hijo2 = padre2
        
        hijos[i] = hijo1
        hijos[i+1] = hijo2

        i = i+2
    
    return hijos