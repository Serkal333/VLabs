import matplotlib.pyplot as plt
import numpy as np
import copy

#   1.1     Начальные значения
variant = 17
np.random.seed(variant)
item_size = 100
otkl = variant/100

x = list (range(1,101))
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

f_y = copy.copy(otn_chast)
for i in range (1, f_y.size):
    f_y[i]=f_y[i-1]+f_y[i]


figure, axis = plt.subplots(2, 4)
axis[0, 0].plot(x, normal)
axis[0, 0].set_title('Title 1')
axis[0, 1].hist(normal, edgecolor = 'black', bins=12)
axis[0, 2].hist(normal, edgecolor = 'black', bins=12)

axis[0, 3].plot(pocket_list, f_y)

plt.tight_layout()
plt.show()
print(f_y)
#print(sum(pocket_value))