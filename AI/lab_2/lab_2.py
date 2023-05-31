import matplotlib.pyplot as plt
import numpy as np

#   Начальные значения
variant = 5
np.random.seed(variant)

# Параметры обучения
delta_C1 = 0.45
teach_speed = 0.3
epochs = int(np.random.uniform(7, 12))
symbol_count = 9
item_size = int(epochs * symbol_count)
#error_chance = np.random.normal(0, 1)
error_chance = 0.2 
batch_size = 7

# Изображения
symbols = [[0,0,1,0,0,1,0,0,1,0,0,1,0,0,1],
            [1,1,1,0,0,1,1,1,1,1,0,0,1,1,1],
            [1,1,1,0,0,1,1,1,1,0,0,1,1,1,1],
            [1,0,1,1,0,1,1,1,1,0,0,1,0,0,1],
            [1,1,1,1,0,0,1,1,1,0,0,1,1,1,1],
            [1,1,1,1,0,0,1,1,1,1,0,1,1,1,1],
            [1,1,1,0,0,1,0,0,1,0,0,1,0,0,1],
            [1,1,1,1,0,1,1,1,1,1,0,1,1,1,1],
            [1,1,1,1,0,1,1,1,1,0,0,1,1,1,1],
            [1,1,1,1,0,1,1,0,1,1,0,1,1,1,1],
            [0,0,0,0,1,0,1,1,1,0,1,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
            [0,0,0,1,0,1,0,1,0,1,0,1,0,0,0],
            [0,0,0,0,0,1,0,1,0,1,0,0,0,0,0],
            [0,1,0,1,0,1,1,1,1,1,0,1,1,0,1],
            [1,1,1,1,0,0,1,0,0,1,0,0,1,0,0],
            [1,1,1,1,0,0,1,1,0,1,0,0,1,1,1],
            [1,0,1,1,0,1,1,1,0,1,0,1,1,0,1],
            [1,0,1,1,0,1,1,1,1,1,0,1,1,0,1],
            [1,1,1,1,0,1,1,0,1,1,0,1,1,0,1],
            [1,1,1,1,0,1,1,1,1,1,0,0,1,0,0],
            [1,1,1,1,0,0,1,0,0,1,0,0,1,1,1],
            [1,1,1,0,1,0,0,1,0,0,1,0,0,1,0],
            [1,0,1,1,0,1,1,1,1,0,0,1,1,1,1],
            [1,0,1,1,0,1,0,1,0,1,0,1,1,0,1],
            [1,0,0,1,0,0,1,1,1,1,0,1,1,1,1],
            [1,1,0,0,0,1,0,1,1,0,0,1,1,1,0],
            [1,1,1,1,0,1,1,0,1,0,1,1,1,0,1],
            [0,0,1,0,0,1,1,1,1,1,0,1,1,1,1],
            [1,1,0,1,0,1,1,0,1,1,0,1,1,1,0],
            [1,0,0,1,0,0,1,1,1,1,0,1,1,0,1],
            [1,1,1,1,0,0,1,1,0,1,0,0,1,0,0],
            [0,1,1,0,1,0,1,1,1,0,1,0,0,1,0],
            [0,0,1,0,0,1,0,0,1,1,0,1,1,1,1],
            [0,1,0,0,0,0,0,1,0,0,1,0,0,1,1],
            [1,0,0,1,0,0,1,0,0,1,0,0,1,1,1],
            [1,1,1,1,0,1,1,0,1,1,1,0,1,0,1],
            [1,0,1,1,0,1,1,0,1,1,0,1,0,1,0],
            [1,0,1,1,0,1,0,1,0,0,1,0,0,1,0],
            [1,1,1,0,0,1,0,1,0,1,0,0,1,1,1]]

class Train:
    def __init__(self):
        # Отбор символов для поиска
        self.sym_prob = np.random.uniform(0, 1, len(symbols))
        # Генерация шумов
        self.noise = np.random.uniform(0, 1, (symbol_count*3, 16))
        # Отбор символов для поиска: индексы (= symbol_count)
        self.symbol_index = []
        for i in range(len(self.sym_prob)):
            if self.sym_prob[i] < 0.2:
                self.symbol_index.append(i)
        if len(self.symbol_index) < symbol_count:
            for i in range(0, 40):
                if i not in self.symbol_index:
                    self.symbol_index.append(i)
                if len(self.symbol_index) > symbol_count:
                    break
        self.symbol_index = self.symbol_index[:symbol_count]
        # Отбор символов для поиска: сами символы
        self.sym = []
        for i in range(0, symbol_count):
            self.sym.append(symbols[self.symbol_index[i]])
            self.sym[i].insert(0, 1)
        # Генерация зашумленных символов
        self.sym_noised = []
        for i in range(0, 3*symbol_count):
            self.sym_noised.append(symbols[self.symbol_index[i%symbol_count]])
            for j in range(0, 16):
                v = self.sym[i%symbol_count][j]
                self.sym_noised[i][j] = v if self.noise[i][j] > error_chance else int(not v)
        # d
        self.d = []
        for i in range(0, symbol_count):
            self.d.append(np.zeros(item_size))
        for i in range(0, symbol_count):
            for j in range(item_size):
                self.d[i][j] = 1 if j % symbol_count == i else -1
        # Веса
        self.w_0 = np.ones(16)
        self.dw = []
        for i in range(0, 16):
            self.dw.append(np.zeros(item_size))

        self.w = []
        for i in range(0, 16):
            self.w.append(np.zeros(item_size))
        # Параметры тренировки
        self.x2 = np.zeros(item_size)
        self.y2 = np.zeros(item_size)
        self.e2 = np.zeros(item_size)
        self.error_count = np.zeros(item_size)
    
    # Первая итерация решения
    def first_iter(self):
        for i in range(0, symbol_count):
            self.x2[i] = np.nanprod(np.dstack((self.sym[i], self.w_0)), 2).sum(1)
            self.y2[i] = 1 if self.x2[i] >= 0 else -1
            self.e2[i] = (self.d[0][i] - self.y2[i]) / 2
            self.error_count[i] = 1 if self.e2[i] != 0 else 0
            for j in range(0, 16):
                self.dw[j][i] = 0 if self.e2[i] == 0 else teach_speed * self.sym[i][j] * self.e2[i]
                if i == 0:
                    self.w[j][0] = self.dw[j][0] + self.w_0[j]
                else:
                    self.w[j][i] = self.dw[j][i] + self.w[j][i-1]

    # Все последующие итерации решения
    def all_iter(self):
        for i in range(symbol_count+1, item_size):
            w = np.zeros(16)
            for j in range(0, 16):
                w[j] = self.w[j][i-1]
            self.x2[i] = np.nanprod(np.dstack((self.sym[i%symbol_count], w)), 2).sum(1)
            self.y2[i] = 1 if self.x2[i] >= 0 else -1
            self.e2[i] = (self.d[0][i] - self.y2[i]) / 2
            self.error_count[i] = 1 if self.e2[i] != 0 else 0
            for j in range(0, 16):
                self.dw[j][i] = 0 if self.e2[i] == 0 else teach_speed * self.sym[i%symbol_count][j] * self.e2[i]
                self.w[j][i] = self.dw[j][i] + self.w[j][i-1]
    
    # Результат тренировки
    def result(self, axis):
        cell_text = []
        col_header = []
        row_header = []
        rows = []
        # Заполнить значения таблицы
        for i in range(0, epochs):
            row = []
            for j in range(0, 16):
                if i*epochs >= item_size:
                    row.append(float("%.1f" % self.w[j][item_size-1]))
                else:
                    row.append(float("%.1f" % self.w[j][i*epochs]))
            rows.append(row)
        # Легенда колонок
        for i in range(0, 16):
            col_header.append('w' + str(i+1))
        # Легенда строк и долбавление текста значениями
        for i in range(0, epochs):
            row_header.append(' ' + str(1+i) + ' ')
            cell_text.append(rows[i])
        # Таблица
        table = axis.table(cellText=cell_text,
                      rowLabels=row_header,
                      colLabels=col_header,
                      loc='center',
                      )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
    
    # Создать тестовую выборку и зашумить ее
    def test(self, axis):
        cell_text = []
        col_header = ['N', 'x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'd1']
        row_header = []
        rows_values = []
        rows_colors = []
        # Заполнить значения таблицы
        for i in range(0, symbol_count*3):
            row_value = []
            row_color = []
            # N
            row_value.append(1 + i)
            row_color.append("#FFFFFF")
            # x0-x15
            for j in range(0, 16):
                v_with_err = self.sym_noised[i%symbol_count][j]
                c = '#ffe294' if v_with_err == 1 else '#FFFFFF'
                row_value.append(v_with_err)
                row_color.append(c)
            # d1
            row_value.append(int(self.d[i%symbol_count][0]))
            row_color.append("#FFFFFF")
            rows_values.append(row_value)
            rows_colors.append(row_color)
        # Легенда строк и долбавление текста значениями
        for i in range(0, symbol_count*3):
            string = str(1+i%symbol_count) + ' (' + str(self.symbol_index[i%symbol_count]) + ')'
            row_header.append(string)
            cell_text.append(rows_values[i])
        # Отрисовка таблицы
        table = axis.table(cellText=cell_text,
                      rowLabels=row_header,
                      colLabels=col_header,
                      cellColours=rows_colors,
                      loc='center')
        # Таблица
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)

    # Вывод нейросети на зашумленной выборке
    def test_result(self, axis):
        cell_text = []
        col_header = []
        row_header = []
        rows = []
        # Заполнить значения таблицы
        for i in range(0, 3*symbol_count):
            row = []
            w_res = np.zeros(16)
            for j in range(0, 16):
                w_res[j] = self.w[j][item_size-1]
            # Прогноз
            v_1 = np.nanprod(np.dstack((w_res, self.sym_noised[i])), 2).sum(1)
            v_1_justified = 1 if int(v_1) >= 0 else -1
            row.append(v_1_justified)
            # Ошибка
            v_2 = int((self.d[0][i] - v_1_justified)/2)
            row.append(v_2)
            # Кол-во
            v_3 = 0 if v_1_justified == self.d[0][i] else 1
            row.append(v_3)    
            rows.append(row)
        # Легенда колонок
        col_header.append('Прогноз')
        col_header.append('Ошибка')
        col_header.append('Кол-во')
        # Легенда строк и долбавление текста значениями
        for i in range(0, 3*symbol_count):
            row_header.append(1+i)
            cell_text.append(rows[i])
        # Таблица
        table = axis.table(cellText=cell_text,
                      rowLabels=row_header,
                      colLabels=col_header,
                      loc='center',
                      )
        table.scale(1.2, 1.2)
        table.auto_set_font_size(False)
        table.set_fontsize(10)

if __name__ == "__main__":
    t = Train()
    t.first_iter()
    t.all_iter()

    figure, axis = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 2, 1]})
    figure.set_figheight(7)
    figure.set_figwidth(14)

    for i in range(0, 3):
        axis[i].axis('off')
    t.result(axis[0])
    t.test(axis[1])
    t.test_result(axis[2])

    plt.tight_layout()
    plt.show()

        