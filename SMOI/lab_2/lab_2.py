import matplotlib.pyplot as plt
import numpy as np
import copy

#   Начальные значения
variant = 7
np.random.seed(variant)
item_size = 100

x = np.arange(1, item_size+1)
p_a = np.random.uniform(0, 1)
p_a_arr = np.zeros(item_size)
for i in range (0, item_size):
    p_a_arr[i] = p_a
noise = np.random.uniform(0, 1, item_size)
event = np.arange(item_size)
for i in range (0, item_size):
    if noise[i] < p_a:
        event[i] = 1
    else:
        event[i] = 0
kopl = copy.copy(event)
for i in range (1, item_size):
    kopl[i] = kopl[i-1] + event[i]

chast = kopl / x
p_i = np.random.uniform(0, 1, 4)
sum_p = sum(p_i)
p_i = p_i / sum_p
p_kopl = copy.copy(p_i)
for i in range (1, 4):
    p_kopl[i] = p_kopl[i] + p_kopl[i-1]

magic_number = np.arange(0, 4)
magic_number[0] = variant*100 + 23
magic_number[1] = magic_number[0] + magic_number[0]/100 
magic_number[2] = magic_number[1] + magic_number[0] % 100/10
magic_number[3] = magic_number[2] + magic_number[0] % 10

znach = np.arange(0, item_size)
for i in range (0, item_size):
    for j in range (0, 4):
        if noise[i] <= p_kopl[j]:
            znach[i] = magic_number[j]
            break

F = []
for i in range (0, 4):
    F.append(np.zeros(item_size, dtype=int))

for i in range (0, 4):
    if znach[0] == magic_number[i]:
        F[i][0] = 1

for i in range(0, 4):
    for j in range (1, item_size):
        if znach[j] == magic_number[i]:
            F[i][j] = F[i][j-1] + 1
        else:
            F[i][j] = F[i][j-1]

f = []
for i in range (0, 4):
    f.append(np.zeros(item_size))

for i in range(0, 4):
    for j in range (0, item_size):
        f[i][j] = F[i][j] / x[j]

#   Лист 2

np.random.seed(variant*2)

p_i = np.random.uniform(0, 1, 3)
sum_p = sum(p_i)
p_i = p_i / sum_p

p_matrix = []
for i in range (0, 3):
    p_matrix.append(np.random.uniform(0, 1, 3))
for i in range (0, 3):
    sum_p = sum(p_matrix[i])
    p_matrix[i] = p_matrix[i] / sum_p

inter_matrix = copy.deepcopy(p_matrix)
for i in range (0, 3):
    inter_matrix[i][1] += inter_matrix[i][0]
    inter_matrix[i][2] += inter_matrix[i][1]

noise = np.random.uniform(0, 1, item_size+1)

sost = np.arange(0, item_size+1)

for i in range (0, item_size+1):
    if noise[i] <= p_i[0]:
        sost[i] = 1
    elif noise[i] <= p_i[1]:
        sost[i] = 2
    else:
        sost[i] = 3

F2 = []
for i in range (0, 3):
    F2.append(np.zeros(item_size+1, dtype=int))

#for i in range (0, 3):
#    if sost[1] == i+1:
#        F2[i][1] = 1

for i in range(0, 3):
    for j in range (1, item_size+1):
        if sost[j] == i+1:
            F2[i][j] = F2[i][j-1] + 1
        else:
            F2[i][j] = F2[i][j-1]

f2 = []
for i in range (0, 3):
    f2.append(np.zeros(item_size+1))

for i in range(0, 3):
    for j in range (1, item_size+1):
        f2[i][j] = F2[i][j] / x[j-1]           


#   Отрисовка

figure, axis = plt.subplots(2, 3)
figure.set_figheight(7)
figure.set_figwidth(14)
axis[0, 0].plot(x, event, label="События")
axis[0, 0].plot(x, chast, label="Частота")
axis[0, 0].plot(x, p_a_arr, label="P(A)")
axis[0, 1].plot(x, znach)
axis[0, 2].plot(x, f[0], label="f1")
axis[0, 2].plot(x, f[1], label="f2")
axis[0, 2].plot(x, f[2], label="f3")
axis[0, 2].plot(x, f[3], label="f4")

axis[1, 0].plot(x, f[0], label="f1")
axis[1, 0].plot(x, f[1], label="f2")
axis[1, 0].plot(x, f[2], label="f3")
axis[1, 1].plot(x, sost[1:])

axis[0, 0].set_title("Равномерное распределение")
axis[0, 0].legend()
axis[0, 1].set_title("Знач.")
axis[0, 2].set_title("f")
axis[0, 2].legend()
axis[1, 0].set_title("f")
axis[1, 0].legend()
axis[1, 1].set_title("Сост.")

plt.show()