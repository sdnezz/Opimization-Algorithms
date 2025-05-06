import numpy as np  # Импорт библиотеки для работы с массивами и случайными числами
import plotly.graph_objects as go  # Импорт для построения 3D-графиков
import plotly.io as pio  # Импорт для сохранения графиков в HTML
from PySide6.QtWebEngineWidgets import QWebEngineView  # Импорт виджета для отображения HTML в GUI
import random  # Импорт модуля для генерации случайных чисел

class BeesAlgorithm:
    def __init__(self, scoutbeecount=200, selectedbeecount=10, bestbeecount=40, selsitescount=12,
                 bestsitescount=4, range_lower=-2.0, range_upper=2.0, range_shrink=0.98,
                 max_iterations=100, max_stagnation=10, convergence_threshold=1e-8):
        # Конструктор класса BeesAlgorithm для инициализации параметров
        self.scoutbeecount = int(scoutbeecount)  # Количество пчел-разведчиков (N_s)
        self.selectedbeecount = int(selectedbeecount)  # Количество пчел на перспективные участки (N_p)
        self.bestbeecount = int(bestbeecount)  # Количество пчел на элитные участки (N_e)
        self.selsitescount = int(selsitescount)  # Количество перспективных участков (e)
        self.bestsitescount = int(bestsitescount)  # Количество элитных участков (m)
        self.range_lower = range_lower  # Нижняя граница поиска (a)
        self.range_upper = range_upper  # Верхняя граница поиска (b)
        self.range_shrink = range_shrink  # Коэффициент сужения радиуса (alpha)
        self.max_iterations = int(max_iterations)  # Максимальное количество итераций (T)
        self.max_stagnation = int(max_stagnation)  # Максимальное количество итераций без улучшения (S)
        self.convergence_threshold = convergence_threshold  # Порог точности (epsilon)

    def get_params(self):
        # Метод для получения текущих параметров в виде словаря
        return {
            "scoutbeecount": self.scoutbeecount,
            "selectedbeecount": self.selectedbeecount,
            "bestbeecount": self.bestbeecount,
            "selsitescount": self.selsitescount,
            "bestsitescount": self.bestsitescount,
            "range_lower": self.range_lower,
            "range_upper": self.range_upper,
            "range_shrink": self.range_shrink,
            "max_iterations": self.max_iterations,
            "max_stagnation": self.max_stagnation,
            "convergence_threshold": self.convergence_threshold
        }

    def set_params(self, params):
        # Метод для установки параметров из словаря
        if "scoutbeecount" in params:
            self.scoutbeecount = int(params["scoutbeecount"])  # Обновление N_s
        if "selectedbeecount" in params:
            self.selectedbeecount = int(params["selectedbeecount"])  # Обновление N_p
        if "bestbeecount" in params:
            self.bestbeecount = int(params["bestbeecount"])  # Обновление N_e
        if "selsitescount" in params:
            self.selsitescount = int(params["selsitescount"])  # Обновление e
        if "bestsitescount" in params:
            self.bestsitescount = int(params["bestsitescount"])  # Обновление m
        if "range_lower" in params:
            self.range_lower = params["range_lower"]  # Обновление a
        if "range_upper" in params:
            self.range_upper = params["range_upper"]  # Обновление b
        if "range_shrink" in params:
            self.range_shrink = params["range_shrink"]  # Обновление alpha
        if "max_iterations" in params:
            self.max_iterations = int(params["max_iterations"])  # Обновление T
        if "max_stagnation" in params:
            self.max_stagnation = int(params["max_stagnation"])  # Обновление S
        if "convergence_threshold" in params:
            self.convergence_threshold = params["convergence_threshold"]  # Обновление epsilon

    def f(self, x, y):
        # Целевая функция (функция Розенброка для примера)
        # f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
        return (1 - x)**2 + 100 * (y - x**2)**2

    class FloatBee:
        def __init__(self, outer):
            # Конструктор класса пчелы (внутренний класс)
            self.minval = [outer.range_lower] * 2  # Нижняя граница для 2D (a, a)
            self.maxval = [outer.range_upper] * 2  # Верхняя граница для 2D (b, b)
            # Шаг 1: Инициализация случайной позиции пчелы
            self.position = np.array([random.uniform(self.minval[n], self.maxval[n]) for n in range(2)])  # p_i = a + (b - a) * r
            self.fitness = outer.f(self.position[0], self.position[1])  # Вычисление пригодности f(p_i)

        def calcfitness(self, outer):
            # Пересчет пригодности для текущей позиции
            self.fitness = outer.f(self.position[0], self.position[1])  # f(p_i) = f(x, y)

        def goto(self, otherpos, range_list, outer):
            # Шаг 4: Локальный поиск — перемещение в окрестность заданной позиции
            self.position = np.array([otherpos[n] + random.uniform(-range_list[n], range_list[n])
                                    for n in range(len(otherpos))])  # p' = c + u, u ~ U(-r, r)
            self.checkposition()  # Ограничение границами
            self.calcfitness(outer)  # Обновление пригодности

        def gotorandom(self, outer):
            # Шаг 5: Глобальный поиск — перемещение в случайную позицию
            self.position = np.array([random.uniform(self.minval[n], self.maxval[n])
                                    for n in range(2)])  # p_i = a + (b - a) * r
            self.checkposition()  # Ограничение границами
            self.calcfitness(outer)  # Обновление пригодности

        def checkposition(self):
            # Ограничение позиции границами пространства поиска
            self.position = np.clip(self.position, self.minval, self.maxval)  # p' = clip(p', a, b)

        def otherpatch(self, bee_list, range_list):
            # Проверка, находится ли пчела в другой области (не пересекается с заданными)
            if not bee_list:
                return True  # Если список пуст, считаем область уникальной
            for curr_bee in bee_list:
                position = curr_bee.position
                if any(abs(self.position[n] - position[n]) > range_list[n] for n in range(2)):  # |p_i,n - p_j,n| > r_n
                    return True  # Участки различны
            return False  # Участки пересекаются

    def plot(self, window):
        # Основной метод выполнения алгоритма с визуализацией
        # Шаг 1: Инициализация популяции пчел
        beecount = self.scoutbeecount + self.selectedbeecount * self.selsitescount + self.bestbeecount * self.bestsitescount  # N = N_s + N_p * e + N_e * m
        swarm = [self.FloatBee(self) for _ in range(beecount)]  # Создание списка пчел
        range_list = [(self.range_upper - self.range_lower) / 8] * 2  # Начальный радиус поиска r = (b - a) / 8
        best_fitness = float('inf')  # Начальное лучшее значение (бесконечность для минимизации)
        best_position = None  # Лучшая позиция
        trajectory = []  # Траектория лучшего решения
        stagnation_counter = 0  # Счетчик стагнации

        for iteration in range(self.max_iterations):
            # Шаг 2: Оценка пригодности и сортировка
            swarm.sort(key=lambda x: x.fitness)  # Сортировка пчел по f(p_i) (меньше — лучше)
            current_best_fitness = swarm[0].fitness  # Текущее лучшее значение
            current_best_position = swarm[0].position.copy()  # Текущая лучшая позиция

            # Шаг 6: Обновление лучшего решения
            if current_best_fitness < best_fitness:  # Если найдено улучшение
                best_fitness = current_best_fitness  # Обновляем лучшее значение
                best_position = current_best_position  # Обновляем лучшую позицию
                stagnation_counter = 0  # Сбрасываем счетчик стагнации
            else:
                stagnation_counter += 1  # Увеличиваем счетчик стагнации

            # Логирование текущего состояния
            window.log_output(f"Итерация {iteration}: x={best_position}, f(x)={best_fitness}")
            trajectory.append(best_position.copy())  # Добавляем позицию в траекторию

            # Шаг 7: Проверка точки останова по точности
            if abs(best_fitness) < self.convergence_threshold:  # |f(x*)| < epsilon
                window.log_output(f"Точка останова: Значение функции стало слишком маленьким (f(x) = {best_fitness})")
                break

            # Шаг 3: Выбор элитных и перспективных участков
            bestsites = [swarm[0]]  # Первый элитный участок — лучшая пчела
            curr_index = 1
            while len(bestsites) < self.bestsitescount and curr_index < len(swarm):  # Выбор m элитных участков
                if swarm[curr_index].otherpatch(bestsites, range_list):  # Проверка уникальности
                    bestsites.append(swarm[curr_index])  # Добавляем в элитные
                curr_index += 1

            selsites = []  # Список перспективных участков
            while len(selsites) < self.selsitescount and curr_index < len(swarm):  # Выбор e перспективных участков
                if (swarm[curr_index].otherpatch(bestsites, range_list) and
                    swarm[curr_index].otherpatch(selsites, range_list)):  # Проверка уникальности
                    selsites.append(swarm[curr_index])  # Добавляем в перспективные
                curr_index += 1

            # Логирование количества участков
            window.log_output(f"Элитных участков: {len(bestsites)}, Перспективных участков: {len(selsites)}")

            # Шаг 7: Проверка точки останова по стагнации
            if stagnation_counter >= self.max_stagnation:  # Если нет улучшений за S итераций
                window.log_output(f"Точка останова: Значение функции стабилизировалось за {stagnation_counter} итераций (f(x) = {best_fitness})")
                range_list = [r * self.range_shrink for r in range_list]  # Сужение радиуса: r = r * alpha
                break

            # Шаг 4: Локальный поиск для элитных участков
            bee_index = 1  # Индекс для назначения пчел
            for best_bee in bestsites:  # Для каждого элитного участка
                best_range = [r * 0.5 for r in range_list]  # Уменьшенный радиус для элитных: r * 0.5
                for _ in range(self.bestbeecount):  # Отправляем N_e пчел
                    if bee_index >= len(swarm):  # Проверка на выход за пределы популяции
                        break
                    if swarm[bee_index] not in bestsites and swarm[bee_index] not in selsites:  # Исключаем уже выбранных
                        swarm[bee_index].goto(best_bee.position, best_range, self)  # Локальный поиск: p' = c + u
                    bee_index += 1

            # Шаг 4: Локальный поиск для перспективных участков
            for sel_bee in selsites:  # Для каждого перспективного участка
                for _ in range(self.selectedbeecount):  # Отправляем N_p пчел
                    if bee_index >= len(swarm):  # Проверка на выход за пределы популяции
                        break
                    if swarm[bee_index] not in bestsites and swarm[bee_index] not in selsites:  # Исключаем уже выбранных
                        swarm[bee_index].goto(sel_bee.position, range_list, self)  # Локальный поиск: p' = c + u
                    bee_index += 1

            # Шаг 5: Глобальный поиск для пчел-разведчиков
            for bee in swarm[bee_index:]:  # Оставшиеся пчелы становятся разведчиками
                bee.gotorandom(self)  # p_i = a + (b - a) * r

        else:
            # Шаг 7: Проверка точки останова по количеству итераций
            window.log_output(f"Точка останова: Достигнуто максимальное количество итераций ({self.max_iterations}) (f(x) = {best_fitness})")

        # Визуализация результатов
        self.plot_trajectory(window, trajectory, best_position)  # Отрисовка траектории

    def plot_trajectory(self, window, trajectory, best_individual):
        # Функция для построения 3D-графика траектории
        x_vals = [ind[0] for ind in trajectory]  # X-координаты траектории
        y_vals = [ind[1] for ind in trajectory]  # Y-координаты траектории
        z_vals = [self.f(x, y) for x, y in trajectory]  # Значения функции

        fig = go.Figure()  # Создание объекта графика

        # Построение поверхности функции
        x1_vals = np.linspace(self.range_lower, self.range_upper, 50)  # Сетка по X
        x2_vals = np.linspace(self.range_lower, self.range_upper, 50)  # Сетка по Y
        X1, X2 = np.meshgrid(x1_vals, x2_vals)  # Создание сетки координат
        Z = np.array([[self.f(x1, x2) for x1, x2 in zip(x1_row, x2_row)] for x1_row, x2_row in zip(X1, X2)])  # Вычисление f(x, y)

        fig.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='viridis', opacity=0.7))  # Добавление поверхности

        # Добавление траектории поиска
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            name="Траектория"
        ))

        # Форматирование значения функции для отображения
        last_value = z_vals[-1]
        last_value_str = "{:.6f}".format(last_value)
        if abs(last_value) < 1e-6:
            last_value_str = "0.000000"  # Упрощение для малых значений

        # Добавление точки оптимума
        fig.add_trace(go.Scatter3d(
            x=[best_individual[0]], y=[best_individual[1]], z=[last_value],
            mode='markers',
            marker=dict(size=6, color='red'),
            name=f"Оптимум (f(x) = {last_value_str})"
        ))

        # Настройка оформления графика
        fig.update_layout(
            title="Пчелиный алгоритм",
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="F(x)"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        # Сохранение графика в HTML
        html_file = "plot.html"
        pio.write_html(fig, file=html_file, auto_open=False)

        # Очистка предыдущих виджетов в layout
        for i in reversed(range(window.graph_layout.count())):
            widget = window.graph_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Загрузка графика в QWebEngineView
        web_view = QWebEngineView()
        web_view.load(f"file:///{html_file}")
        window.graph_layout.addWidget(web_view)