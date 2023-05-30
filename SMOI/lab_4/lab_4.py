import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson
from matplotlib.widgets import Button
import numpy as np
import copy

#   Начальные значения
variant = 10
np.random.seed(variant)

item_size = 100
x = np.arange(1,item_size+1)
noise = np.arange(float(item_size))
for i in range (0, item_size):
    noise[i] = np.random.rand() * (1 + 1) - 1
damping_factor = np.random.uniform(0.6, 0.9)

class Plot():
    def __init__(self):
        self.a = np.random.normal(3, 0.3)
        self.b = np.random.normal(0.1, 0.01)
        self.k = np.random.normal(7, 0.7)
        self.z = np.zeros(item_size)
        self.damping = np.zeros(item_size)

#   Линейный тренд
p1 = Plot()

for i in range (0, item_size):
    p1.z[i] = p1.a + p1.b*x[i] +p1.k*noise[i]
p1.damping[1] = p1.z[0]
for i in range (2, item_size):
    p1.damping[i] = (1 - damping_factor)*p1.z[i-1] + damping_factor*p1.damping[i-1]

#   Синусоидальный тренд
p2 = Plot()
p2.b = np.random.normal(0.3, 0.03)
p2.c = np.random.normal(5, 0.5)
p2.e = np.random.normal(0.15, 0.015)
p2.d = np.random.normal(0.1, 0.01)
p2.k = np.random.normal(30, 3)

for i in range (0, item_size):
    p2.z[i] = p2.a + p2.b*x[i] + p2.k*noise[i] + p2.c * np.sin(p2.e*(x[i] - p2.d))
p2.damping[1] = p2.z[0]
for i in range (2, item_size):
    p2.damping[i] = (1 - damping_factor)*p2.z[i-1] + damping_factor*p2.damping[i-1]

#   Экспоненциальный тренд
p3 = Plot()
p3.b = np.random.normal(0.3, 0.03)
p3.c = np.random.normal(0.1, 0.01)
p3.k = np.random.normal(0.1, 0.01)

for i in range (0, item_size):
    p3.z[i] = p3.a - p3.b*np.exp(-p3.c*x[i]) + p3.k* noise[i]
p3.damping[1] = p3.z[0]
for i in range (2, item_size):
    p3.damping[i] = (1 - damping_factor)*p3.z[i-1] + damping_factor*p3.damping[i-1]

#   Кусочно-линейный тренд
x_part1 = int(np.random.uniform(15, 35))
x_part2 = x_part1 + int(np.random.uniform(15, 35))
x_part3 = x_part2 + int(np.random.uniform(15, 35))

p4 = Plot()
p4.b1 = np.random.normal(0.3, 0.03)
p4.b2 = np.random.normal(0.1, 0.01)
p4.b3 = np.random.normal(0.2, 0.02)
p4.b4 = np.random.normal(0.05, 0.005)
p4.a1 = np.random.normal(3, 0.3)
p4.a2 = p4.a1 + p4.b1 * x_part1 - p4.b2 * x_part1
p4.a3 = p4.a2 + p4.b2 * x_part2 - p4.b3 * x_part2
p4.a4 = p4.a3 + p4.b3 * x_part3 - p4.b4 * x_part3
p4.k = np.random.normal(7, 0.7)

p4.line_part = np.zeros(item_size)
for i in range (0, x_part1):
    p4.line_part[i] = p4.a1 + p4.b1*x[i]
for i in range (x_part1, x_part2):
    p4.line_part[i] = p4.a2 + p4.b2*x[i]
for i in range (x_part2, x_part3):
    p4.line_part[i] = p4.a1 + p4.b1*x[i]
for i in range (x_part3, item_size):
    p4.line_part[i] = p4.a4 + p4.b4*x[i]

for i in range (0, item_size):
    p4.z[i] = p4.line_part[i] + p4.k*noise[i]
p4.damping[1] = p4.z[0]
for i in range (2, item_size):
    p4.damping[i] = (1 - damping_factor)*p4.z[i-1] + damping_factor*p4.damping[i-1]

#   Кусочно-постоянный тренд
p5 = Plot()
p5.k = np.random.normal(7, 0.7)
p5.kus_pos = np.zeros(item_size)

for i in range (0, x_part1):
    p5.kus_pos[i] = p4.a1
for i in range (x_part1, x_part2):
    p5.kus_pos[i] = p4.a2
for i in range (x_part2, x_part3):
    p5.kus_pos[i] = p4.a1
for i in range (x_part3, item_size):
    p5.kus_pos[i] = p4.a4

for i in range (0, item_size):
    p5.z[i] = p5.kus_pos[i] + p5.k*noise[i]
p5.damping[1] = p5.z[0]
for i in range (2, item_size):
    p5.damping[i] = (1 - damping_factor)*p5.z[i-1] + damping_factor*p5.damping[i-1]

#   Линии тренда

# Линейный тренд
p1.zt = np.polyfit (x, p1.z, 1)
p1.p = np.poly1d (p1.zt)
# Синусоидальный тренд
p2.zt = np.polyfit (x, p2.z, 2)
p2.p = np.poly1d (p2.zt)
# Экспонненциальный тренд
p3.zt = np.polyfit (x, p3.z, 3)
p3.p = np.poly1d (p3.zt)
# Кусочно-линейный тренд
p4.zt = np.polyfit (x, p4.z, 2)
p4.p = np.poly1d (p4.zt)
# Кусочно-постоянный тренд
p5.zt1 = np.polyfit (x, p5.z, 1)
p5.p1 = np.poly1d (p5.zt1)
p5.zt2 = np.polyfit (x, p5.z, 5)
p5.p2 = np.poly1d (p5.zt2)

#   Отрисовка
figure, axis = plt.subplots(2, 3)
figure.set_figheight(7)
figure.set_figwidth(14)

axis[0, 0].plot(x, noise, label='Шум')
axis[0, 0].plot(x, p1.z, label='z')
axis[0, 0].plot(x[1:], p1.damping[1:], label='Затухания')
axis[0, 0].plot(x, p1.p(x), linestyle="--", label='Линейная линия тренда')
axis[0, 1].plot(x, p2.z, label='Функция')
axis[0, 1].plot(x[1:], p2.damping[1:], label='Затухания')
axis[0, 1].plot(x, p2.p(x), linestyle="--", label='Полимиальная линия тренда')
axis[0, 2].plot(x, p3.z, label='Функция')
axis[0, 2].plot(x[1:], p3.damping[1:], label='Затухания')
axis[0, 2].plot(x, p3.p(x), linestyle="--", label='Логорифмическая линия тренда')
axis[1, 0].plot(x, noise, label='Шум')
axis[1, 0].plot(x, p4.line_part, label='Кус.лин.')
axis[1, 0].plot(x, p4.z, label='Функция')
axis[1, 0].plot(x[1:], p4.damping[1:], label='Затухания')
axis[1, 0].plot(x, p4.p(x), linestyle="--", label='Полимиальная линия тренда')
axis[1, 1].plot(x, p5.kus_pos, label='Кус.пост.')
axis[1, 1].plot(x, p5.z, label='Функция')
axis[1, 1].plot(x[1:], p5.damping[1:], label='Затухания')
axis[1, 1].plot(x, p5.p1(x), linestyle="--", label='Линейная линия тренда')
axis[1, 1].plot(x, p5.p2(x), linestyle="--", label='Полимиальная линия тренда')

axis[0, 0].set_title('Линейный тренд')
axis[0, 0].legend()
axis[0,1].set_title('Синусоидальный тренд')
axis[0,1].legend()
axis[0, 2].set_title('Экспонненциальный тренд')
axis[0, 2].legend()
axis[1, 0].set_title('Кусочно-линейный тренд')
axis[1, 0].legend()
axis[1, 1].set_title('Кусочно-постоянный тренд')
axis[1, 1].legend()

plt.show()