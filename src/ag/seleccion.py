import numpy as np
import random

def seleccionarPadres(evaluacion, pob, porcentaje_elitismo, nInd):
    k = 5
    num_individuos = round((1 - porcentaje_elitismo)*nInd)

    if(num_individuos % 2 != 0):
        num_individuos += 1
    
    padres = np.zeros((num_individuos, pob.shape[1]))
    for i in range(num_individuos):
        individuos = np.zeros((k, pob.shape[1]))
        fitness = np.zeros(k)
        for j in range(k):
            rand = random.randint(0, pob.shape[0]-1)
            individuos[j] = pob[rand]
            fitness[j] = evaluacion[rand]
        
        index_mejor_individuo = np.argmax(fitness)
        # index_mejor_individuo = np.argmin(fitness)
        padres[i] = individuos[index_mejor_individuo]

    return padres

def seleccionarSiguientePob(pob, hijos, evalu_pob, porcentaje_elitismo):

    # Emparejar fitness con individuos y ordenarlos por fitness en orden descendente
    evalu_pob_con_indices = list(enumerate(evalu_pob))
    evalu_pob_ordenado = sorted(evalu_pob_con_indices, key=lambda x: x[1], reverse=True)
    # evalu_pob_ordenado = sorted(evalu_pob_con_indices, key=lambda x: x[1], reverse=False)
    
    # Calcular el número total de mejores individuos 
    total_ind = len(evalu_pob)
    total_mejores_ind = round(total_ind * porcentaje_elitismo)

    #Comprobamos que padres e hijos sean números pares
    if(total_mejores_ind%2!=0):
        total_mejores_ind=total_mejores_ind-1

    if (len(hijos)%2!=0):
        hijos.pop()
    
    mejores_individuos = []
    for i in range(total_mejores_ind):
        indice, _ = evalu_pob_ordenado[i]
        mejores_individuos.append(pob[indice])

    mejores_individuos = np.array(mejores_individuos)

    poblacion = np.concatenate((mejores_individuos, hijos), axis=0)
    
    return poblacion