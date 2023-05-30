import matplotlib.pyplot as plt
import numpy as np
import copy

#   Начальные значения
variant = 5
np.random.seed(variant)
item_size = 101

iter = np.arange(0, item_size)
a = np.random.normal(0.1, 0.07)
b = np.random.normal(1, 0.2)

noise = np.random.uniform(-1, 1, item_size)

arrp_1 = np.zeros(item_size)

arrp_1[0] = b*noise[0]
for i in range (1, item_size):
    arrp_1[i] = a*arrp_1[i-1] + b*noise[i]

tau = 30
nacf_1 = np.zeros(tau)

for i in range (0, tau):
    nacf_1[i] = np.corrcoef(arrp_1[:item_size-tau], arrp_1[i:item_size-tau+i])[0,1]

shift1 = 3
arrp_3 = np.zeros(item_size)


arrp_3[0] = b*noise[0]
arrp_3[1] = a*arrp_3[0] + b*noise[1]
arrp_3[2] = a*arrp_3[0] + a*arrp_3[1] + b*noise[2]

for i in range (shift1, item_size):
    arrp_3[i] = a*arrp_3[i-1] + a*arrp_3[i-2] + a*arrp_3[i-3] + b*noise[i]

nacf_3 = np.zeros(tau)

for i in range (0, tau):
    nacf_3[i] = np.corrcoef(arrp_3[:item_size-tau], arrp_3[i:item_size-tau+i])[0,1]

shift2 = 5
arrp_5 = np.zeros(item_size)

arrp_5[0] = b*noise[0]
arrp_5[1] = a*arrp_5[0] + b*noise[1]
arrp_5[2] = a*arrp_5[0] + a*arrp_5[1] + b*noise[2]
arrp_5[3] = a*arrp_5[0] + a*arrp_5[1] + a*arrp_5[2] + b*noise[3]
arrp_5[4] = a*arrp_5[0] + a*arrp_5[1] + a*arrp_5[2] + a*arrp_5[3] + b*noise[4]

for i in range (shift2, item_size):
    arrp_5[i] = a*arrp_5[i-1] + a*arrp_5[i-2] + a*arrp_5[i-3] + a*arrp_5[i-4] + a*arrp_5[i-5] + b*noise[i]

nacf_5 = np.zeros(tau)

for i in range (0, tau):
    nacf_5[i] = np.corrcoef(arrp_5[:item_size-tau], arrp_5[i:item_size-tau+i])[0,1]

x = np.arange(-30, 31)
ccf = np.zeros(2*tau+1)

for i in range (0, tau+1):
    ccf[i] = np.corrcoef(arrp_1[i:i+tau], arrp_3[tau:2*tau])[0,1]

for i in range (tau+1, 2*tau+1):
    ccf[i] = np.corrcoef(arrp_1[tau:2*tau], arrp_3[i:i+tau])[0,1]

#   Линии тренда

zt1 = np.polyfit (iter[:tau], nacf_1, 5)
p1 = np.poly1d (zt1)
zt2 = np.polyfit (iter[:tau], nacf_3, 5)
p2 = np.poly1d (zt2)
zt3 = np.polyfit (iter[:tau], nacf_5, 5)
p3 = np.poly1d (zt3)

#   Отрисовка

figure, axis = plt.subplots(2, 4)
figure.set_figheight(7)
figure.set_figwidth(14)

axis[0, 0].plot(iter, arrp_1)
axis[0, 1].plot(iter, arrp_3)
axis[0, 2].plot(iter, arrp_5)
axis[1, 0].plot(iter[:tau], nacf_1)
axis[1, 0].plot(iter[:tau], p1(iter[:tau]))
axis[1, 1].plot(iter[:tau], nacf_3)
axis[1, 1].plot(iter[:tau], p2(iter[:tau]))
axis[1, 2].plot(iter[:tau], nacf_5)
axis[1, 2].plot(iter[:tau], p3(iter[:tau]))
axis[0, 3].plot(iter, noise)
axis[1, 3].plot(x, ccf)

axis[0, 0].set_title("АРСП_1")
axis[0, 1].set_title("АРСП_3")
axis[0, 2].set_title("АРСП_5")
axis[0, 3].set_title("Шум")
axis[1, 0].set_title("НАКФ_1")
axis[1, 1].set_title("НАКФ_3")
axis[1, 2].set_title("НАКФ_5")
axis[1, 3].set_title("ВКФ_1_3")

plt.show()