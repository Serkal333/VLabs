import matplotlib.pyplot as plt
import numpy as np
import copy

#   1.1     Начальные значения
variant = 2128
np.random.seed(variant)

#Параметры обучения
learn_speed = int(np.random.uniform(0.1, 0.9))
N = int(np.random.uniform(7, 12))

