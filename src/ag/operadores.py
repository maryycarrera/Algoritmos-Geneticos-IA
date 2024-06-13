import numpy as np
import random

def cruzar(padres, pc):
    #hijos = cruce_en_un_punto(padres, pc)
    #hijos = cruce_en_dos_puntos(padres, pc)
    hijos = cruce_uniforme(padres, pc)
    #hijos = cruce_por_extension(padres, pc)
    
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

def cruce_en_un_punto(padres, pc):
    punto_cruce = padres.shape[1] // 2
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

def cruce_en_dos_puntos(padres, pc):
    punto_cruce1 = padres.shape[1] // 3
    punto_cruce2 = 2 * punto_cruce1
    hijos = np.zeros(padres.shape)
    i = 0

    while(i<padres.shape[0]):
        padre1 = padres[i]
        padre2 = padres[i+1]
        rand = random.random()

        if(rand < pc):
            hijo1 = np.concatenate((padre1[:punto_cruce1], padre2[punto_cruce1:punto_cruce2], padre1[punto_cruce2:]))
            hijo2 = np.concatenate((padre2[:punto_cruce1], padre1[punto_cruce1:punto_cruce2], padre2[punto_cruce2:]))
        else:
            hijo1 = padre1
            hijo2 = padre2
        
        hijos[i] = hijo1
        hijos[i+1] = hijo2

        i = i+2
    
    return hijos

def cruce_uniforme(padres, pc):
    hijos = np.zeros(padres.shape)
    i = 0

    while(i<padres.shape[0]):
        padre1 = padres[i]
        padre2 = padres[i+1]
        hijo1 = np.zeros(padre1.shape)
        hijo2 = np.zeros(padre2.shape)

        for j in range(padre1.shape[0]):
            rand = random.random()
            if(rand < pc):
                hijo1[j] = padre2[j]
                hijo2[j] = padre1[j]
            else:
                hijo1[j] = padre1[j]
                hijo2[j] = padre2[j]
        
        hijos[i] = hijo1
        hijos[i+1] = hijo2

        i = i+2
    
    return hijos

def cruce_por_extension(padres, pc):
    hijos = np.zeros(padres.shape)
    i = 0

    while(i<padres.shape[0]):
        padre1 = padres[i]
        padre2 = padres[i+1]
        hijo1 = np.zeros(padre1.shape)
        hijo2 = np.zeros(padre2.shape)

        for j in range(padre1.shape[0]):
            rand = random.random()
            if(rand < pc):
                dif = abs(padre1[j] - padre2[j])
                if(padre1[j] > padre2[j]):
                    hijo1[j] = padre1[j] + dif
                    hijo2[j] = padre2[j] - dif
                else:
                    hijo1[j] = padre1[j] - dif
                    hijo2[j] = padre2[j] + dif
                if(j % 2 != 0):
                    hijo1[j] = round(hijo1[j])
                    hijo2[j] = round(hijo2[j])
            else:
                hijo1[j] = padre1[j]
                hijo2[j] = padre2[j]
        
        hijos[i] = hijo1
        hijos[i+1] = hijo2

        i = i+2
    
    return hijos