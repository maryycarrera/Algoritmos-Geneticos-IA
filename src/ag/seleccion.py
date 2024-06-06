import numpy as np
import random

def seleccionarPadres(evaluacion, pob):
    k = 4
    num_individuos = random.randint(k, pob.shape[0]-1) # Seleccionar un número aleatorio de individuos, no sé si debería coincidir con nInd
    padres = np.zeros((num_individuos, pob.shape[1]))
    for i in range(num_individuos):
        individuos = np.zeros((k, pob.shape[1]))
        fitness = np.zeros(k)
        for j in range(k):
            rand = random.randint(0, num_individuos-1)
            individuos[j] = pob[rand]
            fitness[j] = evaluacion[rand]
        
        index_mejor_individuo = np.argmax(fitness)
        padres[i] = individuos[index_mejor_individuo]

    return padres

def seleccionarSiguientePob(pob, hijos, evalu_pob, evalu_hijos):
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