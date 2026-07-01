import numpy as np
import matplotlib.pyplot as plt

class CargaPontual:
    tipo = "pontual"
    def __init__(self, vetor, apoio):
        self.fx = vetor[0]
        self.fy = vetor[1]
        self.mz = vetor[2]

        self.apoio = apoio
        self.x = apoio.x
        self.y = apoio.y
        apoio.lista_cargas.append(self)

class CargaMomento:
    tipo = "momento"
    def __init__(self, vetor, apoio):
        self.fx = vetor[0]
        self.fy = vetor[1]
        self.mz = vetor[2]

        self.apoio = apoio
        self.x = apoio.x
        self.y = apoio.y
        apoio.lista_cargas.append(self)   

class CargaDistribuida:
    tipo = "distribuida"
    def __init__(self, vetor, apoio0, apoio1):
        self.qx = vetor[0]
        self.qy = vetor[1]

        self.x0 = apoio0.x
        self.y0 = apoio0.y
        self.x1 = apoio1.x
        self.y1 = apoio1.y