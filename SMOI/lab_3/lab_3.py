import matplotlib.pyplot as plt
import numpy as np
import copy

#   Начальные значения
variant = 7
np.random.seed(variant)
item_size = 101

iter = np.arange(0, item_size)
a = np.random.normal(0.2, 0.07)
b = np.random.normal(1, 0.2)

noise = np.random.uniform(-1, 1, item_size)

arrp_1 = np.zeros(item_size)

arrp_1[0] = b*noise[0]
for i in range (1, item_size):
    arrp_1[i] = a*arrp_1[i-1] + b*noise[i-1]

tau = 30
nacf_1 = np.zeros(tau)

for i in range (0, tau):
    nacf_1[i] = np.corrcoef(arrp_1[:item_size-tau], arrp_1[i:item_size-tau+i])[0,1]

shift1 = 3
arrp_3 = np.zeros(item_size+shift1)


arrp_3[0] = noise[0]*b*noise[0]
arrp_3[1] = (noise[0]+noise[1])*b*noise[0]
arrp_3[2] = (noise[0]+noise[1]+noise[2])*b*noise[0]

for i in range (0, item_size):
    arrp_3[i+shift1] = a*arrp_3[i+shift1-1] + a*arrp_3[i+shift1-2] + a*arrp_3[i+shift1-3] + b*noise[i]

nacf_3 = np.zeros(tau)

for i in range (0, tau):
    nacf_3[i] = np.corrcoef(arrp_3[shift1:shift1+item_size-tau], arrp_3[shift1+i:shift1+item_size-tau+i])[0,1]

shift2 = 5
arrp_5 = np.zeros(item_size+shift2)

arrp_5[0] = noise[0]*b*noise[0]
arrp_5[1] = (noise[0]+noise[1])*b*noise[0]
arrp_5[2] = (noise[0]+noise[1]+noise[2])*b*noise[0]
arrp_5[3] = (noise[0]+noise[1]+noise[2]+noise[3])*b*noise[0]
arrp_5[4] = (noise[0]+noise[1]+noise[2]+noise[3]+noise[4])*b*noise[0]

for i in range (0, item_size):
    arrp_5[i+shift2] = a*arrp_5[i+shift2-1] + a*arrp_5[i+shift2-2] + a*arrp_5[i+shift2-3] + a*arrp_5[i+shift2-4] + a*arrp_5[i+shift2-5] + b*noise[i]

nacf_5 = np.zeros(tau)

for i in range (0, tau):
    nacf_5[i] = np.corrcoef(arrp_5[shift2:shift2+item_size-tau], arrp_5[shift2+i:shift2+item_size-tau+i])[0,1]

print(nacf_3)