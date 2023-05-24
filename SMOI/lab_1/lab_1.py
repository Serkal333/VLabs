import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.stats import norm, binom, poisson
import numpy as np
import copy

#   Начальные значения
variant = 10
np.random.seed(variant)
item_size = 100


class Plot:
    def __init__(self):
        self.x = list (range(1,101))
        self.y = (np.zeros(item_size))
        self.otkl = variant/100
        self.pocket_count = 12
        self.min: int
        self.max: int
        self.pocket: float
        self.pocket_list: float
        self.pocket_value: int
        self.otn_chast: float
        self.norm_rasp: float
        self.nr: float
        self.f_y: float
        self.teor = 0 
        self.m = 0

    def count_pocket(self):
        self.pocket_value = np.zeros((self.pocket_count+1),dtype=int)
        for i in range (0, item_size):
            for j in range (0, self.pocket_value.size):
                if self.y[i] <= self.pocket_list[j]:
                    self.pocket_value[j]+=1
                    break

    def form_f_y(self):
        self.f_y = copy.copy(self.otn_chast)
        for i in range (1, self.f_y.size):
            self.f_y[i]=self.f_y[i-1]+self.f_y[i]
    
#   Лист 1. Нормальное распределение
if __name__ == "__main__":
    p1 = Plot()
    p1.y = np.random.normal(variant, p1.otkl, item_size)
    p1.pocket_count = 12
    p1.min = p1.y.min()
    p1.max = p1.y.max()
    p1.pocket = (p1.max-p1.min)/p1.pocket_count
    p1.pocket_list = np.zeros(p1.pocket_count+1)
    p1.pocket_list[0] = p1.min
    for i in range (1, p1.pocket_count+1):
            p1.pocket_list[i] = p1.pocket_list[i-1] + p1.pocket

    p1.count_pocket()
    p1.otn_chast = copy.copy(p1.pocket_value)/item_size

    p1.norm_rasp = np.arange(float(p1.pocket_count+1))*0
    for i in range (0, p1.norm_rasp.size):
        p1.norm_rasp[i] = norm.pdf(p1.pocket_list[i], variant, p1.otkl)
    p1.nr = copy.copy(p1.norm_rasp)/sum(p1.norm_rasp)

    p1.form_f_y()
        
    #   Лист 2. Равномерное распределение
    p2 = Plot()
    p2.y = np.random.uniform(variant, variant*2, item_size)
    p2.pocket = 1
    p2.pocket_count = variant
    p2.min = variant+1
    p2.max = variant*2
    p2.pocket_list = list (range(p2.min, p2.max+1))
    p2.count_pocket()
    p2.pocket_value = p2.pocket_value[:variant]
    p2.otn_chast = copy.copy(p2.pocket_value)/item_size
    p2.teor = (copy.copy(p2.pocket_value)*0+1/variant)
    p2.form_f_y()

    #   Лист 3. Бернулли

    p3 = Plot()
    p3.y = p3.y[:item_size]
    for i in range (0, item_size):
        p3.y = np.random.binomial(1, 0.17, item_size)
    p3.pocket_list = [0, 1]
    p3.pocket_value = np.zeros((2),dtype=int)
    p3.pocket_count = 2
    for i in range (0, item_size):
        if p3.y[i] == 0:
            p3.pocket_value[0]+=1
        else:
            p3.pocket_value[1]+=1
    p3.otn_chast = copy.copy(p3.pocket_value)/item_size
    p3.teor = [1-p3.otkl, p3.otkl]
    p3.f_y = copy.copy(p3.otn_chast)
    p3.f_y[1] = p3.f_y[1]+p3.f_y[0]

    #   Лист 4. Бинауральное

    p4 = Plot()
    p4.m = 5
    p4.y = p4.y[:item_size]
    for i in range (0, item_size):
        p4.y = np.random.binomial(p4.m, 0.17, item_size)
    p4.pocket = 1
    p4.pocket_count = p4.m
    p4.min = 0
    p4.max = p4.m
    p4.pocket_list = list(range(p4.min, p4.max+1))
    p4.count_pocket()
    p4.otn_chast = copy.copy(p4.pocket_value)/item_size

    p4.teor = np.arange(float(p4.pocket_count+1))*0
    for i in range (0, p4.m):
        p4.teor[i] = binom.pmf(p4.pocket_list[i], p4.m, p4.otkl)
    p4.form_f_y()

    #   Лист 5. Пуассона

    p5 = Plot()
    p5.y = np.random.poisson(variant, item_size)
    p5.min = 0
    p5.max = max(p5.y)
    p5.pocket = 1
    p5.pocket_count = p5.max
    p5.pocket_list = list(range(p5.min, p5.max+1))
    p5.count_pocket()
    p5.otn_chast = copy.copy(p5.pocket_value)/item_size
    p5.teor = np.arange(float(p5.pocket_count+1))*0
    for i in range (0, p5.pocket_count):
        p5.teor[i] = poisson.pmf(p5.pocket_list[i], sum(p5.y)/item_size)
    p5.form_f_y()

    # Лист 6. сглаживание

    item_size2 = 200
    rand_num = np.random.rand(1, 10)
    d1 = variant
    d2 = variant * 2
    d3 = variant * 3
    d4 = 200 - variant
    e1 = rand_num.max()
    e2 = np.mean(rand_num)
    e3 = rand_num.min()
    p6 = Plot()
    p6.x = list(range(1,201))

    p6.y = np.random.rand(item_size2)
    msa_d1 = np.zeros(item_size2-d1)
    msa_d2 = np.zeros(item_size2-d2)
    msa_d3 = np.zeros(item_size2-d3)
    msa_d4 = np.zeros(item_size2-d4)
    ecsp_e1 = np.zeros(item_size2-1)
    ecsp_e2 = np.zeros(item_size2-1)
    ecsp_e3 = np.zeros(item_size2-1)

    for i in range (0, item_size2):
        if i + 1 >= d1:
            msa_d1[i-d1] = 0
            for j in range (0, d1):
                msa_d1[i-d1] += p6.y[i-j]
            msa_d1[i-d1] /= d1
        if i + 1 >= d2:
            msa_d2[i-d2] = 0
            for j in range (0, d2):
                msa_d2[i-d2] += p6.y[i-j]
            msa_d2[i-d2] /= d2
        if i + 1 >= d3:
            msa_d3[i-d3] = 0
            for j in range (0, d3):
                msa_d3[i-d3] += p6.y[i-j]
            msa_d3[i-d3] /= d3
        if i + 1 >= d4:
            msa_d4[i-d4] = 0
            for j in range (0, d4):
                msa_d4[i-d4] += p6.y[i-j]
            msa_d4[i-d4] /= d4
        if i == 1:
            ecsp_e1[i-1] = p6.y[i] * e1
            ecsp_e2[i-1] = p6.y[i] * e2
            ecsp_e3[i-1] = p6.y[i] * e3
        if i > 1:
            ecsp_e1[i-1] = p6.y[i] * e1 + (1 - e1) * ecsp_e1[i-2]
            ecsp_e2[i-1] = p6.y[i] * e2 + (1 - e2) * ecsp_e2[i-2]
            ecsp_e3[i-1] = p6.y[i] * e3 + (1 - e3) * ecsp_e3[i-2]

    average = np.zeros(8)
    dispers = np.zeros(8)
    sko = np.zeros(8)

    average[0] = np.mean(p6.y)
    average[1] = np.mean(msa_d1)
    average[2] = np.mean(msa_d2)
    average[3] = np.mean(msa_d3)
    average[4] = np.mean(msa_d4)
    average[5] = np.mean(ecsp_e1)
    average[6] = np.mean(ecsp_e2)
    average[7] = np.mean(ecsp_e3)

    square_deviation = lambda x : (x - np.mean(p6.y)) ** 2 
    dispers[0] = sum( map(square_deviation, p6.y) ) / (item_size2 - 1)
    square_deviation = lambda x : (x - np.mean(msa_d1)) ** 2 
    dispers[1] = sum( map(square_deviation, msa_d1) ) / (item_size2 - 1)
    square_deviation = lambda x : (x - np.mean(msa_d2)) ** 2 
    dispers[2] = sum( map(square_deviation, msa_d2) ) / (item_size2 - 1)
    square_deviation = lambda x : (x - np.mean(msa_d3)) ** 2 
    dispers[3] = sum( map(square_deviation, msa_d3) ) / (item_size2 - 1)
    square_deviation = lambda x : (x - np.mean(msa_d4)) ** 2 
    dispers[4] = sum( map(square_deviation, msa_d4) ) / (item_size2 - 1)
    square_deviation = lambda x : (x - np.mean(ecsp_e1)) ** 2 
    dispers[5] = sum( map(square_deviation, ecsp_e1) ) / (item_size2 - 1)
    square_deviation = lambda x : (x - np.mean(ecsp_e2)) ** 2 
    dispers[6] = sum( map(square_deviation, ecsp_e2) ) / (item_size2 - 1)
    square_deviation = lambda x : (x - np.mean(ecsp_e3)) ** 2 
    dispers[7] = sum( map(square_deviation, ecsp_e3) ) / (item_size2 - 1)

    for i in range (0,8):
        sko[i] = dispers[i] ** (0.5)
    
    #print(p5.teor)
    

    #   Отрисовка

    def draw_page(page):
        match page:
            case 0:
                axis[0, 0].plot(p1.x, p1.y)
                axis[0, 1].hist(p1.y, edgecolor = 'black', bins=p1.pocket_count)
                axis[0, 2].hist(p1.y,  weights=np.ones_like (p1.y) / len (p1.y), edgecolor = 'black', bins=p1.pocket_count)
                axis[0, 2].plot(p1.pocket_list, p1.nr)
                axis[0, 3].plot(p1.pocket_list, p1.f_y)
                axis[1, 0].plot(p2.x, p2.y)
                axis[1, 1].hist(p2.y, edgecolor = 'black', bins=p2.pocket_count)
                axis[1, 2].hist(p2.y,  weights=np.ones_like (p2.y) / len (p2.y), edgecolor = 'black', bins=p2.pocket_count)
                axis[1, 2].plot(p2.pocket_list, p2.teor)
                axis[1, 3].plot(p2.pocket_list, p2.f_y)
                
                axis[0, 0].set_title('Нормальное распределение')
                axis[0, 1].set_title('Гистограмма')
                axis[0, 2].set_title('Гистограмма')
                axis[0, 3].set_title('F(y)')
                axis[1, 0].set_title('Равномерное распределение')
                axis[1, 1].set_title('Гистограмма')
                axis[1, 2].set_title('Гистограмма')
                axis[1, 3].set_title('F(y)')

            case 1:
                axis[0, 0].plot(p3.x, p3.y)
                axis[0, 1].hist(p3.y, edgecolor = 'black', bins=p3.pocket_count)
                axis[0, 2].hist(p3.y, weights=np.ones_like (p3.y) / len (p3.y), edgecolor = 'black', bins=p3.pocket_count)
                axis[0, 2].plot(p3.pocket_list, p3.teor)
                axis[0, 3].plot(p3.pocket_list, p3.f_y)
                axis[1, 0].plot(p4.x, p4.y)
                axis[1, 1].hist(p4.y, edgecolor = 'black', bins=p4.pocket_count)
                axis[1, 2].hist(p4.y, weights=np.ones_like (p4.y) / len (p4.y), edgecolor = 'black', bins=p4.pocket_count)
                axis[1, 2].plot(p4.pocket_list, p4.teor)
                axis[1, 3].plot(p4.pocket_list, p4.f_y)

                axis[0, 0].set_title('Бернулли')
                axis[0, 1].set_title('Гистограмма')
                axis[0, 2].set_title('Гистограмма')
                axis[0, 3].set_title('F(y)')
                axis[1, 0].set_title('Биноминальное')
                axis[1, 1].set_title('Гистограмма')
                axis[1, 2].set_title('Гистограмма')
                axis[1, 3].set_title('F(y)')
                
            case 2:
                axis[0, 0].plot(p5.x, p5.y)
                axis[0, 1].hist(p5.y, edgecolor = 'black', bins=p5.pocket_count)
                axis[0, 2].hist(p5.y, weights=np.ones_like (p5.y) / len (p5.y), edgecolor = 'black', bins=p5.pocket_count)
                axis[0, 2].plot(p5.pocket_list, p5.teor)
                axis[0, 3].plot(p5.pocket_list, p5.f_y)

                axis[1, 0].plot(p6.x, p6.y, label="Фактический")
                axis[1, 0].plot(list(range(d1+1,201)), msa_d1, label="Сглаживание d1")
                axis[1, 0].plot(list(range(d2+1,201)), msa_d2, label="Сглаживание d2")
                axis[1, 0].plot(list(range(d3+1,201)), msa_d3, label="Сглаживание d3")
                axis[1, 0].plot(list(range(d4+1,201)), msa_d4, label="Сглаживание d4")
                
                axis[1, 1].plot(p6.x, p6.y, label="Фактический")
                axis[1, 1].plot(list(range(2,201)), ecsp_e1, label="Прогноз e1")
                axis[1, 1].plot(list(range(2,201)), ecsp_e2, label="Прогноз e2")
                axis[1, 1].plot(list(range(2,201)), ecsp_e3, label="Прогноз e3")

                axis[1, 2].plot(1, average[0], ':o', label="Среднее СП")
                axis[1, 2].plot(1, dispers[0], ':o', label="Дисп. СП")
                axis[1, 2].plot(1, sko[0], ':o', label="СКО СП")
                axis[1, 2].plot(list(range(2, 6)), np.array([average[i] for i in range (1, 5)]), label="Среднее МСС")
                axis[1, 2].plot(list(range(2, 6)), np.array([dispers[i] for i in range (1, 5)]), label="Дисп. МСС")
                axis[1, 2].plot(list(range(2, 6)), np.array([sko[i] for i in range (1, 5)]), label="СКО МСС")
                axis[1, 2].plot(list(range(6, 9)), np.array([average[i] for i in range (5, 8)]), label="Среднее МЭС")
                axis[1, 2].plot(list(range(6, 9)), np.array([dispers[i] for i in range (5, 8)]), label="Дисп. МЭС")
                axis[1, 2].plot(list(range(6, 9)), np.array([sko[i] for i in range (5, 8)]), label="СКО МЭС")


                axis[0, 0].set_title('Распределение Пуассона')
                axis[0, 1].set_title('Гистограмма')
                axis[0, 2].set_title('Гистограмма')
                axis[0, 3].set_title('F(y)')
                axis[1, 0].set_title('Метод скользящего среднего')
                axis[1, 0].legend()
                axis[1, 1].set_title('Экспоненциальное сглаживание')
                axis[1, 1].legend()
                axis[1, 2].set_title('Статистические характеристики сглаженных CП')
                axis[1, 2].legend()
                
    freqs = 3
    fig, axis = plt.subplots(2, 4)
    fig.subplots_adjust(bottom=0.2)
    draw_page(0)
    

    class Index:
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % freqs
            for k in range (0, 2):
                for j in range (0, 4):
                    axis[k, j].clear()
            draw_page(i)
            plt.draw()

        def prev(self, event):
            self.ind -= 1
            i = self.ind % freqs
            for k in range (0, 2):
                for j in range (0, 4):
                    axis[k, j].clear()
            draw_page(i)
            plt.draw()

    callback = Index()
    axprev = fig.add_axes([0.7, 0.02, 0.1, 0.075])
    axnext = fig.add_axes([0.81, 0.02, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    #plt.tight_layout()
    plt.show()

