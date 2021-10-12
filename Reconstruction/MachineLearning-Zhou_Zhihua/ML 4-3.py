import numpy as np
import math
def Ent(D):
    if len(D)==0:
        return 0
    entro=0
    for k in D:
        entro=entro-k*math.log(k,2)
    return entro

def TreeGenerate(D,A):
    
    