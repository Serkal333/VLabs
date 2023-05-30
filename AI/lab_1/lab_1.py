import matplotlib.pyplot as plt
import numpy as np
import copy

#   Начальные значения
variant = 5
np.random.seed(variant)

# Параметры обучения
delta_C1 = np.random.uniform(0.2, 0.8)  
teach_speed = np.random.uniform(0.1, 0.9)
epochs = int(np.random.uniform(7, 12))
epoch_size = int(np.random.uniform(15, 25))
item_size = int(epochs * epoch_size)

delta_C1 = 0.45 
teach_speed = 0.35

class Train:
    def __init__(self):

        #   Начальные значения

        self.x1_c1 = np.random.uniform(0, 10)
        self.x2_c1 = np.random.uniform(5, 15)
        self.x3_c1 = np.random.uniform(0, 10)
        self.x4_c1 = np.random.uniform(5, 15)
        self.x5_c1 = np.random.uniform(0, 10)
        self.sko_c1 = np.random.uniform(1, 3)
        self.x1_c2 = np.random.uniform(0, 10)
        self.x2_c2 = np.random.uniform(5, 15)
        self.x3_c2 = np.random.uniform(0, 10)
        self.x4_c2 = np.random.uniform(5, 15)
        self.x5_c2 = np.random.uniform(0, 10)
        self.sko_c2 = np.random.uniform(1, 3)

        #   Случайные числа
        
        self.r1 = np.random.normal(0, 1, item_size)
        self.r2 = np.random.normal(0, 1, item_size)
        self.r3 = np.random.normal(0, 1, item_size)
        self.r4 = np.random.normal(0, 1, item_size)
        self.r5 = np.random.normal(0, 1, item_size)
        self.sigma = np.random.sample(item_size)
        
        self.s_wx = np.zeros(item_size)
        self.y_predict = np.zeros(item_size)
        self.e_error = np.zeros(item_size)
        self.e_count = np.zeros(item_size)

        self.learn_d = np.zeros(item_size)
        self.learn_x0 = np.ones(item_size)
        self.learn_x1 = np.zeros(item_size)
        self.learn_x2 = np.zeros(item_size)
        self.learn_x3 = np.zeros(item_size)
        self.learn_x4 = np.zeros(item_size)
        self.learn_x5 = np.zeros(item_size)
        self.x1_class1 = np.zeros(item_size)
        self.x2_class1 = np.zeros(item_size)
        self.x1_class2 = np.zeros(item_size)
        self.x2_class2 = np.zeros(item_size)

        self.dw0 = np.zeros(item_size)
        self.dw1 = np.zeros(item_size)
        self.dw2 = np.zeros(item_size)
        self.dw3 = np.zeros(item_size)
        self.dw4 = np.zeros(item_size)
        self.dw5 = np.zeros(item_size)

        self.w0 = np.zeros(item_size)
        self.w1 = np.zeros(item_size)
        self.w2 = np.zeros(item_size)
        self.w3 = np.zeros(item_size)
        self.w4 = np.zeros(item_size)
        self.w5 = np.zeros(item_size)

        self.epochs_err_count = np.zeros(epochs+1)
        self.epochs_w0 = np.zeros(epochs+1)
        self.epochs_w1 = np.zeros(epochs+1)
        self.epochs_w2 = np.zeros(epochs+1)
        self.epochs_w3 = np.zeros(epochs+1)
        self.epochs_w4 = np.zeros(epochs+1)
        self.epochs_w5 = np.zeros(epochs+1)
        self.y1_graph = 0
        self.y2_graph = 0

    #   Батчи для листов 2, 3

    def make_batch(self):
        t1 = self.r1[0:epoch_size]
        for i in range(0, epochs):
            for j in range(0, epoch_size):
                self.r1[j+i*epoch_size] = t1[j]
        t2 = self.r2[0:epoch_size]
        for i in range(0, epochs):
            for j in range(0, epoch_size):
                self.r2[j+i*epoch_size] = t2[j]
        t3 = self.sigma[0:epoch_size]
        for i in range(0, epochs):
            for j in range(0, epoch_size):
                self.sigma[j+i*epoch_size] = t3[j]
        t5 = self.learn_d[0:epoch_size]
        for i in range(0, epochs):
            for j in range(0, epoch_size):
                self.learn_d[j+i*epoch_size] = t5[j]

    #   Установка пераметров для листа 5

    def setup_list_5(self):
        self.x1_c1 = 0
        self.x1_c2 = 0
        self.x2_c1 = 5
        self.x2_c2 = 5
        self.sko_c1 = 1
        self.sko_c2 = 1

    #   Обучающая выборка
    
    def fill_learn_data(self):  
        for i in range(0, self.learn_d.size):
            self.learn_d[i] = 1 if self.sigma[i] < delta_C1 else -1
        for i in range(0, self.learn_x1.size):
            self.learn_x1[i] = (self.x1_c1 + self.sko_c1 * self.r1[i]) if self.learn_d[i] == 1 else (self.x1_c2 + self.sko_c2 * self.r1[i])
        for i in range(0, self.learn_x2.size):
            self.learn_x2[i] = (self.x2_c1 + self.sko_c1 * self.r2[i]) if self.learn_d[i] == 1 else (self.x2_c2 + self.sko_c2 * self.r2[i])
        for i in range(0, self.learn_x2.size):
            self.learn_x3[i] = (self.x3_c1 + self.sko_c1 * self.r3[i]) if self.learn_d[i] == 1 else (self.x3_c2 + self.sko_c2 * self.r3[i])
        for i in range(0, self.learn_x2.size):
            self.learn_x4[i] = (self.x4_c1 + self.sko_c1 * self.r4[i]) if self.learn_d[i] == 1 else (self.x4_c2 + self.sko_c2 * self.r4[i])
        for i in range(0, self.learn_x2.size):
            self.learn_x5[i] = (self.x5_c1 + self.sko_c1 * self.r5[i]) if self.learn_d[i] == 1 else (self.x5_c2 + self.sko_c2 * self.r5[i])

    #   Два класса

    def specify_class(self):
        for i in range(0, item_size):
            self.x1_class1[i] = self.learn_x1[i] if self.learn_d[i] == 1 else -100
        for i in range(0, item_size):
            self.x2_class1[i] = self.learn_x2[i] if self.learn_d[i] == 1 else -100
        for i in range(0, item_size):
            self.x1_class2[i] = self.learn_x1[i] if self.learn_d[i] == -1 else -100
        for i in range(0, item_size):
            self.x2_class2[i] = self.learn_x2[i] if self.learn_d[i] == -1 else -100

    #   Обучение перцептрона
    
    def teach(self, isList3 = False):

        #   Первая итерация
        
        w_first = [0.1, 0.1, 0.1]
        self.s_wx[0] = np.nanprod(np.dstack((np.array([self.learn_x0[0], self.learn_x1[0], self.learn_x2[0]]), w_first)), 2).sum(1)
        self.y_predict[0] = 1 if self.s_wx[0] >= self.sigma[0] else -1
        self.e_error[0] = (self.y_predict[0]-self.learn_d[0])/2
        self.e_count[0] = 1 if self.e_error[0] != 0 else 0
        self.dw0[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x0[0]
        self.dw1[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x1[0]
        self.dw2[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x2[0]
        self.w0[0] = w_first[0] + self.dw0[0] if not isList3 else 0
        self.w1[0] = w_first[1] + self.dw1[0]
        self.w2[0] = w_first[2] + self.dw2[0]

        #   Все итерации

        for i in range(1, item_size):
            self.s_wx[i] = np.nanprod(np.dstack((np.array([self.learn_x0[i], self.learn_x1[i], self.learn_x2[i]]), np.array([self.w0[i-1], self.w1[i-1], self.w2[i-1]]))), 2).sum(1)
            self.y_predict[i] = 1 if self.s_wx[i] >= self.sigma[i] else -1
            self.e_error[i] = (self.learn_d[i]-self.y_predict[i])/2
            self.e_count[i] = 1 if self.e_error[i] != 0 else 0
            self.dw0[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x0[i]
            self.dw1[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x1[i]
            self.dw2[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x2[i]
            self.w0[i] = self.w0[i-1] + self.dw0[i] if not isList3 else 0
            self.w1[i] = self.w1[i-1] + self.dw1[i]
            self.w2[i] = self.w2[i-1] + self.dw2[i]

    #   Обучение для листа 4 (5 классов)

    def teach_list4(self):

        #   Первая итерация

        w_first = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.s_wx[0] = np.nanprod(np.dstack((
            np.array([self.learn_x0[0], self.learn_x1[0], self.learn_x2[0], self.learn_x3[0], self.learn_x4[0], self.learn_x5[0]]),
            w_first)), 2).sum(1)
        self.y_predict[0] = 1 if self.s_wx[0] >= self.sigma[0] else -1
        self.e_error[0] = (self.y_predict[0]-self.learn_d[0])/2
        self.e_count[0] = 1 if self.e_error[0] != 0 else 0
        self.dw0[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x0[0]
        self.dw1[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x1[0]
        self.dw2[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x2[0]
        self.dw3[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x3[0]
        self.dw4[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x4[0]
        self.dw5[0] = 0 if self.e_error[0] == 0 else teach_speed * self.e_error[0] * self.learn_x5[0]
        self.w0[0] = w_first[0] + self.dw0[0]
        self.w1[0] = w_first[1] + self.dw1[0]
        self.w2[0] = w_first[2] + self.dw2[0]
        self.w3[0] = w_first[3] + self.dw3[0]
        self.w4[0] = w_first[4] + self.dw4[0]
        self.w5[0] = w_first[5] + self.dw5[0]

        #   Все итерации

        for i in range(1, item_size):
            self.s_wx[i] = np.nanprod(np.dstack((
                np.array([self.learn_x0[i], self.learn_x1[i], self.learn_x2[i], self.learn_x3[i], self.learn_x4[i], self.learn_x5[i]]), 
                np.array([self.w0[i-1], self.w1[i-1], self.w2[i-1], self.w3[i-1], self.w4[i-1], self.w5[i-1]]))), 2).sum(1)
            self.y_predict[i] = 1 if self.s_wx[i] >= self.sigma[i] else -1
            self.e_error[i] = (self.learn_d[i]-self.y_predict[i])/2
            self.e_count[i] = 1 if self.e_error[i] != 0 else 0
            self.dw0[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x0[i]
            self.dw1[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x1[i]
            self.dw2[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x2[i]
            self.dw3[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x3[i]
            self.dw4[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x4[i]
            self.dw5[i] = 0 if self.e_error[i] == 0 else teach_speed * self.e_error[i] * self.learn_x5[i]
            self.w0[i] = self.w0[i-1] + self.dw0[i]
            self.w1[i] = self.w1[i-1] + self.dw1[i]
            self.w2[i] = self.w2[i-1] + self.dw2[i]
            self.w3[i] = self.w3[i-1] + self.dw3[i]
            self.w4[i] = self.w4[i-1] + self.dw4[i]
            self.w5[i] = self.w5[i-1] + self.dw5[i]

    #   Результаты обучения

    def results(self):
        for i in range(0, epochs+1):
            self.epochs_err_count[i] = sum(self.e_count[i*epoch_size:(i+1)*epoch_size])
            self.epochs_w0[i] = self.w0[i*(epoch_size-1)]
            self.epochs_w1[i] = self.w1[i*(epoch_size-1)]
            self.epochs_w2[i] = self.w2[i*(epoch_size-1)]
        self.y1_graph = ( self.epochs_w0[epochs]/self.epochs_w2[epochs] - self.epochs_w1[epochs]/self.epochs_w2[epochs]) * (-5)
        self.y2_graph = (-self.epochs_w0[epochs]/self.epochs_w2[epochs] - self.epochs_w1[epochs]/self.epochs_w2[epochs]) * (15)

    #   Результаты обучения для листа 3 (w0 = 0)

    def results_list3(self):
        for i in range(0, epochs+1):
            self.epochs_err_count[i] = sum(self.e_count[i*epoch_size:(i+1)*epoch_size])
            self.epochs_w0[i] = 0
            self.epochs_w1[i] = self.w1[i*(epoch_size-1)]
            self.epochs_w2[i] = self.w2[i*(epoch_size-1)]
        self.y1_graph = ( self.epochs_w0[epochs]/self.epochs_w2[epochs] - self.epochs_w1[epochs]/self.epochs_w2[epochs]) * (-5)
        self.y2_graph = (-self.epochs_w0[epochs]/self.epochs_w2[epochs] - self.epochs_w1[epochs]/self.epochs_w2[epochs]) * (15)

    #   Результаты обучения для листа 4 (5 классов)

    def results_list4(self):
        for i in range(0, epochs+1):
            self.epochs_err_count[i] = sum(self.e_count[i*epoch_size:(i+1)*epoch_size])
            self.epochs_w0[i] = self.w0[i*(epoch_size-1)]
            self.epochs_w1[i] = self.w1[i*(epoch_size-1)]
            self.epochs_w2[i] = self.w2[i*(epoch_size-1)]
            self.epochs_w3[i] = self.w3[i*(epoch_size-1)]
            self.epochs_w4[i] = self.w4[i*(epoch_size-1)]
            self.epochs_w5[i] = self.w5[i*(epoch_size-1)]

def set_dimensions(x, y):
    return [x, y] if x < 10 and y < 50 and x > 1 and y > 1 else [-5, 20]

def test(str):
    print(str)

if __name__ == "__main__":
    t1 = Train()
    t2 = copy.deepcopy(t1)
    t3 = copy.deepcopy(t1)
    t4 = copy.deepcopy(t1)
    t5 = copy.deepcopy(t1)

    #   Обучение

    #   Лист 1
    t1.fill_learn_data()
    t1.specify_class()
    t1.teach()
    t1.results()
    #   Лист 2
    t2.make_batch()
    t2.fill_learn_data()
    t2.specify_class()
    t2.teach()
    t2.results()
    #   Лист 3
    t3.make_batch()
    t3.fill_learn_data()
    t3.specify_class()
    t3.teach(True)
    t3.results_list3()
    #   Лист 4
    t4.fill_learn_data()
    t4.specify_class()
    t4.teach_list4()
    t4.results_list4()
    #   Лист 5
    t5.setup_list_5()
    t5.fill_learn_data()
    t5.specify_class()
    t5.teach()
    t5.results()

    #   Отрисовка

    figure, axis = plt.subplots(2, 5)
    figure.set_figheight(7)
    figure.set_figwidth(14)

    axis[0, 0].scatter(t1.x1_class1, t1.x2_class1, s=5, label='С1')
    axis[0, 0].scatter(t1.x1_class2, t1.x2_class2, s=5, label='С2')
    axis[0, 0].plot([-5, 15], [t1.y1_graph, t1.y2_graph])
    axis[0, 0].set_xlim(-5, 20)
    axis[0, 0].set_ylim(set_dimensions(t1.y1_graph, t1.y2_graph))

    axis[0, 1].scatter(t2.x1_class1, t2.x2_class1, s=5, label='С1')
    axis[0, 1].scatter(t2.x1_class2, t2.x2_class2, s=5, label='С2')
    axis[0, 1].plot([-5, 15], [t2.y1_graph, t2.y2_graph])
    axis[0, 1].set_xlim(-5, 20)
    axis[0, 1].set_ylim(set_dimensions(t2.y1_graph, t2.y2_graph))
    
    axis[0, 2].scatter(t3.x1_class1, t3.x2_class1, s=5, label='C1')
    axis[0, 2].scatter(t3.x1_class2, t3.x2_class2, s=5, label='C2')
    axis[0, 2].plot([-5, 15], [t3.y1_graph, t3.y2_graph])
    axis[0, 2].set_xlim(-5, 20)
    axis[0, 2].set_ylim(set_dimensions(t3.y1_graph, t3.y2_graph))

    axis[0, 3].plot(np.arange(epochs), t4.epochs_w0[:epochs], label='w0')
    axis[0, 3].plot(np.arange(epochs), t4.epochs_w1[:epochs], label='w1')
    axis[0, 3].plot(np.arange(epochs), t4.epochs_w2[:epochs], label='w2')
    axis[0, 3].plot(np.arange(epochs), t4.epochs_w3[:epochs], label='w3')
    axis[0, 3].plot(np.arange(epochs), t4.epochs_w4[:epochs], label='w4')
    axis[0, 3].plot(np.arange(epochs), t4.epochs_w5[:epochs], label='w5')
    axis[0, 3].set_xlim(0, epochs-1)
    axis[0, 3].set_ylim(np.amin([t4.epochs_w0, t4.epochs_w1, t4.epochs_w2, t4.epochs_w3, t4.epochs_w4, t4.epochs_w5])-1,
                        np.amax([t4.epochs_w0, t4.epochs_w1, t4.epochs_w2, t4.epochs_w3, t4.epochs_w4, t4.epochs_w5])+1)

    axis[0, 4].scatter(t5.x1_class1, t5.x2_class1, s=5, label='C1')
    axis[0, 4].scatter(t5.x1_class2, t5.x2_class2, s=5, label='C2')
    axis[0, 4].plot([-5, 15], [t5.y1_graph, t5.y2_graph])
    axis[0, 4].set_xlim(-5, 20)
    axis[0, 4].set_ylim(set_dimensions(t5.y1_graph, t5.y2_graph))

    axis[1, 0].plot(np.arange(epochs), t1.epochs_err_count[:epochs], label='Ошибка')
    axis[1, 0].plot(np.arange(epochs), t1.epochs_w0[:epochs], label='w0')
    axis[1, 0].plot(np.arange(epochs), t1.epochs_w1[:epochs], label='w1')
    axis[1, 0].plot(np.arange(epochs), t1.epochs_w2[:epochs], label='w2')
    axis[1, 0].set_xlim(0, epochs-1)

    axis[1, 1].plot(np.arange(epochs), t2.epochs_err_count[:epochs], label='Ошибка')
    axis[1, 1].plot(np.arange(epochs), t2.epochs_w0[:epochs], label='w0')
    axis[1, 1].plot(np.arange(epochs), t2.epochs_w1[:epochs], label='w1')
    axis[1, 1].plot(np.arange(epochs), t2.epochs_w2[:epochs], label='w2')
    axis[1, 1].set_xlim(0, epochs-1)

    axis[1, 2].plot(np.arange(epochs), t3.epochs_err_count[:epochs], label='Ошибка')
    axis[1, 2].plot(np.arange(epochs), t3.epochs_w0[:epochs], label='w0')
    axis[1, 2].plot(np.arange(epochs), t3.epochs_w1[:epochs], label='w1')
    axis[1, 2].plot(np.arange(epochs), t3.epochs_w2[:epochs], label='w2')
    axis[1, 2].set_xlim(0, epochs-1)

    axis[1, 3].plot(np.arange(epochs), t4.epochs_err_count[:epochs], label='Ошибка')
    axis[1, 3].set_xlim(0, epochs-1)
    axis[1, 3].set_ylim(0, np.amax(t4.epochs_err_count))
    
    axis[1, 4].plot(np.arange(epochs), t5.epochs_err_count[:epochs], label='Ошибка')
    axis[1, 4].plot(np.arange(epochs), t5.epochs_w0[:epochs], label='w0')
    axis[1, 4].plot(np.arange(epochs), t5.epochs_w1[:epochs], label='w1')
    axis[1, 4].plot(np.arange(epochs), t5.epochs_w2[:epochs], label='w2')
    axis[1, 4].set_xlim(0, epochs-1)

    axis[0, 0].set_title('x1 x2')
    axis[0, 0].legend()
    axis[0, 1].set_title('x1 x2 с батчами')
    axis[0, 1].legend()
    axis[0, 2].set_title('x1 x2 с батчами без w0')
    axis[0, 2].legend()
    axis[0, 3].set_title('x1 x2 x3 x4 x5')
    axis[0, 3].legend()
    axis[0, 4].set_title('x1 = x2, c1 = c2')
    axis[0, 4].legend()
    axis[1, 0].set_title('Сходимость')
    axis[1, 0].legend()
    axis[1, 1].set_title('Сходимость')
    axis[1, 1].legend()
    axis[1, 2].set_title('Сходимость')
    axis[1, 2].legend()
    axis[1, 3].legend()
    axis[1, 3].set_title('Сходимость')
    axis[1, 4].set_title('Сходимость')
    axis[1, 4].legend()

    plt.tight_layout()
    plt.show()