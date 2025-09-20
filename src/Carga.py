import numpy as np
import matplotlib.pyplot as plt

class CargaPontual:
    tipo = "pontual"
    def __init__(self, vetor, posicao):
        self.fx = vetor[0]
        self.fy = vetor[1]
        self.mz = vetor[2]

        self.x = posicao[0]
        self.y = posicao[1]

class CargaDistribuida:
    tipo = "distribuida"
    def __init__(self, vetor, posicao0, posicao1):
        self.qx = vetor[0]
        self.qy = vetor[1]

        self.x0 = posicao0[0]
        self.y0 = posicao0[1]

        self.x1 = posicao1[0]
        self.y1 = posicao1[1]
