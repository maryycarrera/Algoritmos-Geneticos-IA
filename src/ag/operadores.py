import numpy as np
import random

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

def mutar(pob, pm):
    i = 0

    while(i<pob.shape[0]):
        individuo = pob[i]
        rand = random.random()

        if(rand < pm):
            gen = random.randint(0, individuo.shape[0]-1)
            individuo[gen] = 1 - individuo[gen]

        i = i+1

    return pob