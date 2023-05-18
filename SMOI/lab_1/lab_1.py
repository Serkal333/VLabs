import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import copy

#   Начальные значения
variant = 17
np.random.seed(variant)
item_size = 100
x = list (range(1,101))

#   Лист 1. Нормальное распределение
otkl = variant/100
normal = np.random.normal(variant, otkl, item_size)


pocket_count = 12
min = normal.min()
max = normal.max()
pocket = (max-min)/pocket_count

pocket_list = np.arange(float(pocket_count+1))
pocket_list[0] = min
for i in range (1, 13):
    pocket_list[i] = pocket_list[i-1] + pocket
pocket_value = np.arange(pocket_count+1)*0
for i in range (0, item_size):
    for j in range (0, pocket_value.size):
        if normal[i] <= pocket_list[j]:
            pocket_value[j]+=1
            break

#otn_chast = np.arange(float(pocket_count+1)*0)
otn_chast = copy.copy(pocket_value)/item_size

norm_rasp = np.arange(float(pocket_count+1))*0
for i in range (0, norm_rasp.size):
    norm_rasp[i] = norm.pdf(pocket_list[i], variant, otkl)

nr = copy.copy(norm_rasp)/sum(norm_rasp)

f_y = copy.copy(otn_chast)
for i in range (1, f_y.size):
    f_y[i]=f_y[i-1]+f_y[i]

#   Лист 2. Равномерное распределение

ravn = np.random.uniform(variant, variant*2, item_size)


#   Отрисовка

figure, axis = plt.subplots(2, 4)
axis[0, 0].plot(x, normal)
axis[0, 0].set_title('Title 1')
axis[0, 1].hist(normal, edgecolor = 'black', bins=12)
axis[0, 2].hist(normal,  weights=np.ones_like (normal) / len (normal), edgecolor = 'black', bins=12)
axis[0, 2].plot(pocket_list, nr)
axis[0, 3].plot(pocket_list, f_y)

plt.tight_layout()
plt.show()
print(nr)
