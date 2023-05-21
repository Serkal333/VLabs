import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.stats import norm, binom, poisson
import numpy as np
import copy

#   Начальные значения
variant = 17
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
        p5.teor[i] = poisson.pmf(p5.pocket_list[i], sum(p5.y)/item_size) #!!!!!!!!!!ХЗ насколько это правильно
    p5.form_f_y()

    # Лист 6. сглаживание

    
    print(p5.teor)
    

    #   Отрисовка

    def draw_page(page):
        match page:
            case 0:
                axis[0, 0].plot(p1.x, p1.y)
                axis[0, 1].hist(p1.y, edgecolor = 'black', bins=p1.pocket_count)
                axis[0, 2].hist(p1.y,  weights=np.ones_like (p1.y) / len (p1.y), edgecolor = 'black', bins=p1.pocket_count)
                axis[0, 2].plot(p1.pocket_list, p1.nr)
                axis[0, 3].plot(p1.pocket_list, p1.f_y)
                axis[1, 0].plot(p1.x, p1.y)
                axis[1, 1].hist(p2.y, edgecolor = 'black', bins=p1.pocket_count)
                axis[1, 2].hist(p2.y,  weights=np.ones_like (p2.y) / len (p2.y), edgecolor = 'black', bins=variant)
                axis[1, 2].plot(p2.pocket_list, p2.teor)
                axis[1, 3].plot(p2.pocket_list, p2.f_y)
                axis[0, 0].set_title('Title 1')
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
                
            case 2:
                axis[0, 0].plot(p5.x, p5.y)
                axis[0, 1].hist(p5.y, edgecolor = 'black', bins=p5.pocket_count)
                axis[0, 2].hist(p5.y, weights=np.ones_like (p5.y) / len (p5.y), edgecolor = 'black', bins=p5.pocket_count)
                axis[0, 2].plot(p5.pocket_list, p5.teor)
                axis[0, 3].plot(p5.pocket_list, p5.f_y)
    
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

    plt.tight_layout()
    plt.show()

