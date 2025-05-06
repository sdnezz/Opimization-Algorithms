import numpy as np  # Импорт библиотеки для работы с массивами и случайными числами
import plotly.graph_objects as go  # Импорт библиотеки Plotly для построения графиков (контурные графики и траектории)
import plotly.io as pio  # Импорт Plotly для сохранения графиков в HTML-формате
from PySide6.QtWidgets import QVBoxLayout, QWidget  # Импорт виджетов PySide6 для создания GUI (макет и виджет-контейнер)
from PySide6.QtWebEngineWidgets import QWebEngineView  # Импорт виджета PySide6 для отображения HTML-графиков в GUI

class ImmuneNetworkOptimization:
    def __init__(self):
        # Конструктор класса ImmuneNetworkOptimization для инициализации параметров алгоритма
        self.initial_point = [1.0, 1.0]  # Начальная точка (x, y) для старта поиска (в данном случае не используется напрямую)
        self.max_iterations = 100  # Максимальное количество итераций алгоритма (T)
        self.pop_size = 100  # Размер начальной популяции антител (|S_b|)
        self.n_b = 20  # Число лучших антител, которые будут клонироваться на каждой итерации
        self.n_c = 10  # Максимальное число клонов, создаваемых для одного антитела
        self.b_s = 0.1  # Доля лучших клонов, сохраняемых после мутации (в долях от общего числа клонов)
        self.b_b = 0.01  # Порог аффинности для отбора клонов (клоны с аффинностью < b_b отбрасываются)
        self.b_r = 0.02  # Порог расстояния для сжатия сети (если расстояние между антителами меньше b_r, одно из них удаляется)
        self.b_n = 0.1  # Доля антител, заменяемых случайными на каждой итерации (для поддержания разнообразия)
        self.mutation_rate = 0.5  # Сила мутации (насколько сильно изменяются координаты антитела при мутации)
        self.range_lower = -2  # Нижняя граница пространства поиска (a)
        self.range_upper = 2  # Верхняя граница пространства поиска (b)
        self.f = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2  # Целевая функция (Розенброка), минимум в (1, 1) с f(1, 1) = 0

    class Antibody:
        def __init__(self, x, y, outer):
            # Конструктор класса Antibody (представляет одно антитело — решение)
            self.x = x  # Координата x антитела
            self.y = y  # Координата y антитела
            self.bg_affinity = 1 / (1 + outer.f(x, y))  # Вычисление аффинности (bg_affinity), обратной к значению функции: bg_affinity = 1 / (1 + f(x, y))

    def get_params(self):
        # Метод для получения текущих параметров алгоритма в виде словаря
        return {
            "initial_point": self.initial_point,  # Начальная точка
            "max_iterations": self.max_iterations,  # Максимальное количество итераций
            "pop_size": self.pop_size,  # Размер популяции
            "n_b": self.n_b,  # Число лучших антител для клонирования
            "n_c": self.n_c,  # Максимальное число клонов
            "b_s": self.b_s,  # Доля сохраняемых клонов
            "b_b": self.b_b,  # Порог аффинности для клонов
            "b_r": self.b_r,  # Порог расстояния для сжатия
            "b_n": self.b_n,  # Доля заменяемых антител
            "mutation_rate": self.mutation_rate,  # Сила мутации
            "range_lower": self.range_lower,  # Нижняя граница поиска
            "range_upper": self.range_upper  # Верхняя граница поиска
        }

    def set_params(self, params):
        # Метод для установки параметров алгоритма из словаря
        if "initial_point" in params:
            self.initial_point = params["initial_point"]  # Обновление начальной точки
        if "max_iterations" in params:
            self.max_iterations = params["max_iterations"]  # Обновление максимального числа итераций
        if "pop_size" in params:
            self.pop_size = params["pop_size"]  # Обновление размера популяции
        if "n_b" in params:
            self.n_b = params["n_b"]  # Обновление числа лучших антител для клонирования
        if "n_c" in params:
            self.n_c = params["n_c"]  # Обновление максимального числа клонов
        if "b_s" in params:
            self.b_s = params["b_s"]  # Обновление доли сохраняемых клонов
        if "b_b" in params:
            self.b_b = params["b_b"]  # Обновление порога аффинности
        if "b_r" in params:
            self.b_r = params["b_r"]  # Обновление порога расстояния
        if "b_n" in params:
            self.b_n = params["b_n"]  # Обновление доли заменяемых антител
        if "mutation_rate" in params:
            self.mutation_rate = params["mutation_rate"]  # Обновление силы мутации
        if "range_lower" in params:
            self.range_lower = params["range_lower"]  # Обновление нижней границы
        if "range_upper" in params:
            self.range_upper = params["range_upper"]  # Обновление верхней границы

    def compute_bb_affinity(self, ab1, ab2):
        # Метод для вычисления ВВ-аффинности (евклидова расстояния между двумя антителами)
        # Формула: sqrt((x1 - x2)^2 + (y1 - y2)^2)
        return np.sqrt((ab1.x - ab2.x)**2 + (ab1.y - ab2.y)**2)  # Возвращает расстояние между ab1 и ab2

    def initialize_population(self):
        # Метод для создания начальной популяции из pop_size случайных антител
        # Шаг 1: Инициализация популяции
        return [self.Antibody(np.random.uniform(self.range_lower, self.range_upper),  # Генерация случайного x в [range_lower, range_upper]
                              np.random.uniform(self.range_lower, self.range_upper),  # Генерация случайного y в [range_lower, range_upper]
                              self)  # Передача ссылки на внешний объект для доступа к f(x, y)
                for _ in range(self.pop_size)]  # Создание списка из pop_size антител

    def clone_antibody(self, antibody):
        # Метод для клонирования антитела, число клонов пропорционально аффинности
        # Шаг 4: Клонирование антитела
        num_clones = int(1 + (self.n_c - 1) * antibody.bg_affinity)  # Число клонов: 1 + (n_c - 1) * bg_affinity (от 1 до n_c)
        return [self.Antibody(antibody.x, antibody.y, self) for _ in range(num_clones)]  # Создание списка из num_clones копий антитела

    def mutate_antibody(self, antibody):
        # Метод для мутации антитела: изменение координат с учетом mutation_rate
        # Шаг 5: Мутация антитела
        x_new = antibody.x + self.mutation_rate * np.random.uniform(-0.5, 0.5)  # Новый x: x + mutation_rate * U(-0.5, 0.5)
        y_new = antibody.y + self.mutation_rate * np.random.uniform(-0.5, 0.5)  # Новый y: y + mutation_rate * U(-0.5, 0.5)
        x_new = np.clip(x_new, self.range_lower, self.range_upper)  # Ограничение x границами [range_lower, range_upper]
        y_new = np.clip(y_new, self.range_lower, self.range_upper)  # Ограничение y границами [range_lower, range_upper]
        return self.Antibody(x_new, y_new, self)  # Создание нового антитела с мутированными координатами

    def run(self):
        # Основной метод выполнения иммунного сетевого алгоритма
        # Шаг 1: Инициализация популяции
        S_b = self.initialize_population()  # Создание начальной популяции S_b из pop_size антител
        S_m = []  # Инициализация пустого списка памяти S_m для хранения лучших решений
        best_solution = None  # Переменная для хранения лучшего решения (антитела)
        trajectory = []  # Список для хранения траектории лучшего решения
        iterations_log = []  # Список для логов итераций
        fitness_history = []  # Список для истории значений функции лучшего решения
        stagnation_count = 0  # Счетчик стагнации (для критерия остановки по стабильности)
        epsilon = 1e-6  # Порог для критерия остановки по стабильности (малое значение разницы значений функции)

        # Основной цикл алгоритма
        for iteration in range(self.max_iterations):  # Цикл по итерациям (от 0 до max_iterations - 1)
            # Шаг 2: Оценка пригодности (уже выполнена при создании антител в Antibody)
            # Шаг 3: Отбор лучших антител
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)  # Сортировка S_b по аффинности (по убыванию)
            selected = S_b[:self.n_b]  # Выбор n_b лучших антител с наибольшей аффинностью

            # Шаг 4: Клонирование
            clones = []  # Список для хранения всех клонов
            for ab in selected:  # Для каждого из выбранных антител
                clones.extend(self.clone_antibody(ab))  # Создание клонов и добавление их в список
            # Шаг 5: Мутация клонов
            clones = [self.mutate_antibody(clone) for clone in clones]  # Мутация каждого клона

            # Шаг 6: Отбор лучших клонов
            clones = sorted(clones, key=lambda ab: ab.bg_affinity, reverse=True)  # Сортировка клонов по аффинности (по убыванию)
            n_d = int(self.b_s * len(clones))  # Вычисление числа сохраняемых клонов: b_s * |clones|
            new_memory = clones[:n_d]  # Выбор n_d лучших клонов
            new_memory = [ab for ab in new_memory if ab.bg_affinity >= self.b_b]  # Фильтрация клонов по порогу аффинности b_b
            # Шаг 7: Добавление в память
            S_m.extend(new_memory)  # Добавление отобранных клонов в память S_m

            # Шаг 8: Сжатие памяти (удаление похожих антител в S_m)
            i = 0  # Индекс для перебора антител в S_m
            while i < len(S_m):  # Пока не обработаны все антитела
                j = i + 1  # Индекс для сравнения с другими антителами
                while j < len(S_m):  # Перебор всех антител после i
                    if self.compute_bb_affinity(S_m[i], S_m[j]) < self.b_r:  # Если расстояние между антителами меньше b_r
                        del S_m[j]  # Удаляем одно из антител (j-е)
                    else:
                        j += 1  # Переходим к следующему антителу
                i += 1  # Переходим к следующему i

            # Шаг 9: Объединение популяции
            S_b.extend(S_m)  # Добавление антител из памяти S_m в основную популяцию S_b

            # Шаг 10: Сжатие сети (удаление похожих антител в S_b)
            i = 0  # Индекс для перебора антител в S_b
            while i < len(S_b):  # Пока не обработаны все антитела
                j = i + 1  # Индекс для сравнения
                while j < len(S_b):  # Перебор всех антител после i
                    if self.compute_bb_affinity(S_b[i], S_b[j]) < self.b_r:  # Если расстояние меньше b_r
                        del S_b[j]  # Удаляем одно из антител (j-е)
                    else:
                        j += 1  # Переходим к следующему антителу
                i += 1  # Переходим к следующему i

            # Шаг 11: Замена худших антител
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)  # Сортировка S_b по аффинности
            num_replace = int(self.b_n * self.pop_size)  # Вычисление числа заменяемых антител: b_n * pop_size
            S_b = S_b[:self.pop_size - num_replace]  # Удаление num_replace худших антител
            S_b.extend(self.initialize_population()[:num_replace])  # Добавление num_replace случайных антител

            # Шаг 12: Ограничение популяции
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)[:self.pop_size]  # Сортировка и обрезка до pop_size

            # Шаг 13: Обновление лучшего решения
            current_best = S_b[0]  # Текущее лучшее антитело (с наибольшей аффинностью)
            current_fitness = self.f(current_best.x, current_best.y)  # Значение функции для текущего лучшего антитела
            if best_solution is None or current_fitness < self.f(best_solution.x, best_solution.y):  # Если улучшение
                best_solution = current_best  # Обновляем лучшее решение
                iterations_log.append(f"Улучшение на итерации {iteration}: x=[{best_solution.x:.6f}, {best_solution.y:.6f}], f(x)={current_fitness:.6f}")  # Логируем улучшение
            else:
                iterations_log.append(f"Итерация {iteration}: x=[{best_solution.x:.6f}, {best_solution.y:.6f}], f(x)={current_fitness:.6f}")  # Логируем текущую итерацию

            # Шаг 14: Проверка критериев остановки
            fitness_history.append(current_fitness)  # Добавление текущего значения функции в историю
            if len(fitness_history) > 10:  # Если в истории больше 10 значений
                fitness_history.pop(0)  # Удаляем самое старое значение
                max_diff = max(abs(fitness_history[i] - fitness_history[i-1]) for i in range(1, len(fitness_history)))  # Максимальная разница между соседними значениями
                if max_diff < epsilon:  # Если разница меньше порога epsilon (стабильность)
                    stagnation_count += 1  # Увеличиваем счетчик стагнации
                else:
                    stagnation_count = 0  # Сбрасываем счетчик стагнации
                if stagnation_count >= 10:  # Если значение функции стабильно 10 итераций
                    iterations_log.append(f"Остановка: значение функции стабильно в течение 10 итераций (разница < {epsilon})")  # Логируем остановку
                    break  # Прерываем цикл

            if current_fitness < 1e-6:  # Если значение функции меньше порога (достигнут минимум)
                iterations_log.append("Достигнут минимум!")  # Логируем достижение минимума
                break  # Прерываем цикл

            trajectory.append([best_solution.x, best_solution.y])  # Добавляем текущую лучшую позицию в траекторию

        final_point = [best_solution.x, best_solution.y]  # Финальная точка (координаты лучшего решения)
        return final_point, trajectory, "Иммунная сеть завершена", iterations_log  # Возвращаем результат: финальную точку, траекторию, сообщение и логи

    def plot(self, window):
        # Метод для выполнения алгоритма и визуализации результатов
        final_point, trajectory, message, iterations_log = self.run()  # Выполнение алгоритма и получение результатов

        for log in iterations_log:  # Перебор логов итераций
            window.log_output(log)  # Вывод каждого лога в интерфейс
        window.log_output(f"Финальная точка: x=[{final_point[0]:.6f}, {final_point[1]:.6f}], f(x)={self.f(*final_point):.6f}")  # Вывод финальной точки и значения функции
        window.log_output(message)  # Вывод сообщения о завершении

        # Построение контурного графика функции
        x = np.linspace(self.range_lower, self.range_upper, 100)  # Сетка координат x (100 точек от range_lower до range_upper)
        y = np.linspace(self.range_lower, self.range_upper, 100)  # Сетка координат y (100 точек от range_lower до range_upper)
        X, Y = np.meshgrid(x, y)  # Создание сетки координат (X, Y) для построения контура
        Z = np.array([[self.f(x_i, y_i) for x_i, y_i in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])  # Вычисление значений функции f(x, y) для каждой точки сетки

        fig = go.Figure()  # Создание объекта графика Plotly
        fig.add_trace(go.Contour(  # Добавление контурного графика функции
            z=Z, x=x, y=y,  # Указание значений Z (f(x, y)), X и Y
            colorscale='Viridis',  # Цветовая схема (Viridis)
            contours=dict(showlabels=True),  # Отображение меток на контурах
            showscale=True  # Отображение цветовой шкалы
        ))
        traj_x, traj_y = zip(*trajectory)  # Разделение траектории на списки x и y координат
        fig.add_trace(go.Scatter(  # Добавление траектории поиска
            x=traj_x, y=traj_y,  # Координаты траектории
            mode='lines+markers',  # Режим отображения: линии и маркеры
            line=dict(color='red', width=2),  # Линии красного цвета, толщина 2
            marker=dict(size=6),  # Маркеры размером 6
            name='Траектория'  # Название траектории в легенде
        ))
        fig.add_trace(go.Scatter(  # Добавление точки минимума
            x=[final_point[0]], y=[final_point[1]],  # Координаты финальной точки
            mode='markers',  # Режим отображения: только маркеры
            marker=dict(size=10, color='blue', symbol='star'),  # Маркер: синяя звезда размером 10
            name='Минимум'  # Название точки в легенде
        ))

        fig.update_layout(  # Настройка оформления графика
            title="Траектория иммунной сети",  # Заголовок графика
            xaxis_title="x",  # Название оси X
            yaxis_title="y",  # Название оси Y
            showlegend=True,  # Отображение легенды
            margin=dict(l=0, r=0, b=0, t=30)  # Настройка отступов
        )

        html_file = "plot.html"  # Имя файла для сохранения графика
        pio.write_html(fig, file=html_file, auto_open=False)  # Сохранение графика в HTML-файл без автоматического открытия

        for i in reversed(range(window.graph_layout.count())):  # Очистка предыдущих виджетов в layout
            widget = window.graph_layout.itemAt(i).widget()  # Получение виджета по индексу
            if widget:  # Если виджет существует
                widget.setParent(None)  # Удаление виджета из layout

        web_view = QWebEngineView()  # Создание виджета для отображения HTML-графика
        web_view.load(f"file:///{html_file}")  # Загрузка HTML-файла в виджет
        window.graph_layout.addWidget(web_view)  # Добавление виджета в layout интерфейса