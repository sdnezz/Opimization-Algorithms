import numpy as np  # Импорт библиотеки NumPy для числовых вычислений
import plotly.graph_objects as go  # Импорт Plotly для создания графиков
import plotly.io as pio  # Импорт Plotly IO для сохранения графиков
from PySide6.QtWidgets import QVBoxLayout, QWidget  # Импорт виджетов PySide6 для интерфейса
from PySide6.QtWebEngineWidgets import QWebEngineView  # Импорт QWebEngineView для отображения HTML-графиков

class ImmuneNetworkOptimization:  # Определение класса для алгоритма иммунной сети
    def __init__(self):  # Инициализация класса
        self.initial_point = [1.0, 1.0]  # Начальная точка [x, y], задаётся через интерфейс
        self.max_iterations = 200  # Максимальное число итераций алгоритма
        self.pop_size = 100  # Размер популяции антител
        self.n_b = 20  # Число лучших антител для клонирования
        self.n_c = 10  # Максимальное число клонов для одного антитела
        self.b_s = 0.1  # Доля лучших клонов для сохранения
        self.b_b = 0.01  # Порог BG-аффинности для отбора клонов
        self.b_r = 0.05  # Порог BB-аффинности для сжатия сети
        self.b_n = 0.15  # Доля антител, заменяемых случайными
        self.mutation_rate = 0.5  # Начальная сила мутации
        self.range_lower = -2  # Нижняя граница пространства поиска
        self.range_upper = 2  # Верхняя граница пространства поиска
        self.f = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2  # Функция Розенброка

    class Antibody:  # Внутренний класс для представления антитела
        def __init__(self, x, y, outer):  # Инициализация антитела
            self.x = x  # Координата x антитела
            self.y = y  # Координата y антитела
            self.bg_affinity = 1 / (1 + outer.f(x, y))  # Вычисление BG-аффинности: 1 / (1 + f(x, y))

    def get_params(self):  # Метод для получения параметров алгоритма
        return {  # Возвращает словарь с текущими параметрами
            "initial_point": self.initial_point,  # Начальная точка
            "max_iterations": self.max_iterations,  # Максимальное число итераций
            "pop_size": self.pop_size,  # Размер популяции
            "n_b": self.n_b,  # Число лучших антител
            "n_c": self.n_c,  # Максимальное число клонов
            "b_s": self.b_s,  # Доля сохраняемых клонов
            "b_b": self.b_b,  # Порог BG-аффинности
            "b_r": self.b_r,  # Порог BB-аффинности
            "b_n": self.b_n,  # Доля заменяемых антител
            "mutation_rate": self.mutation_rate,  # Сила мутации
            "range_lower": self.range_lower,  # Нижняя граница поиска
            "range_upper": self.range_upper  # Верхняя граница поиска
        }

    def set_params(self, params):  # Метод для установки параметров из интерфейса
        if "initial_point" in params:  # Проверка наличия initial_point
            self.initial_point = params["initial_point"]  # Установка начальной точки
        if "max_iterations" in params:  # Проверка наличия max_iterations
            self.max_iterations = params["max_iterations"]  # Установка числа итераций
        if "pop_size" in params:  # Проверка наличия pop_size
            self.pop_size = params["pop_size"]  # Установка размера популяции
        if "n_b" in params:  # Проверка наличия n_b
            self.n_b = params["n_b"]  # Установка числа лучших антител
        if "n_c" in params:  # Проверка наличия n_c
            self.n_c = params["n_c"]  # Установка числа клонов
        if "b_s" in params:  # Проверка наличия b_s
            self.b_s = params["b_s"]  # Установка доли сохраняемых клонов
        if "b_b" in params:  # Проверка наличия b_b
            self.b_b = params["b_b"]  # Установка порога BG-аффинности
        if "b_r" in params:  # Проверка наличия b_r
            self.b_r = params["b_r"]  # Установка порога BB-аффинности
        if "b_n" in params:  # Проверка наличия b_n
            self.b_n = params["b_n"]  # Установка доли заменяемых антител
        if "mutation_rate" in params:  # Проверка наличия mutation_rate
            self.mutation_rate = params["mutation_rate"]  # Установка силы мутации
        if "range_lower" in params:  # Проверка наличия range_lower
            self.range_lower = params["range_lower"]  # Установка нижней границы
        if "range_upper" in params:  # Проверка наличия range_upper
            self.range_upper = params["range_upper"]  # Установка верхней границы

    def compute_bb_affinity(self, ab1, ab2):  # Метод для вычисления BB-аффинности
        return np.sqrt((ab1.x - ab2.x)**2 + (ab1.y - ab2.y)**2)  # Евклидово расстояние между антителами

    def initialize_population(self):  # Метод для создания начальной популяции
        return [self.Antibody(np.random.uniform(self.range_lower, self.range_upper),  # Создание антитела с случайным x
                              np.random.uniform(self.range_lower, self.range_upper), self)  # Случайный y и ссылка на outer
                for _ in range(self.pop_size)]  # Повтор для pop_size антител

    def clone_antibody(self, antibody):  # Метод для клонирования антитела
        num_clones = int(1 + (self.n_c - 1) * antibody.bg_affinity)  # Число клонов, зависит от BG-аффинности
        return [self.Antibody(antibody.x, antibody.y, self) for _ in range(num_clones)]  # Создание списка клонов

    def mutate_antibody(self, antibody, current_mutation_rate):  # Метод для мутации антитела
        x_new = antibody.x + current_mutation_rate * np.random.uniform(-0.5, 0.5)  # Случайное изменение x
        y_new = antibody.y + current_mutation_rate * np.random.uniform(-0.5, 0.5)  # Случайное изменение y
        x_new = np.clip(x_new, self.range_lower, self.range_upper)  # Ограничение x границами поиска
        y_new = np.clip(y_new, self.range_lower, self.range_upper)  # Ограничение y границами поиска
        return self.Antibody(x_new, y_new, self)  # Создание нового антитела с новыми координатами

    def run(self):  # Основной метод выполнения алгоритма
        S_b = self.initialize_population()  # Инициализация популяции антител
        S_m = []  # Инициализация пустой памяти для клонов
        best_solution = None  # Лучшее решение, изначально None
        trajectory = []  # Список для хранения траектории лучших точек
        iterations_log = []  # Список для логов итераций
        fitness_history = []  # Список для истории значений f(x, y)
        stagnation_count = 0  # Счётчик итераций с малым изменением f(x, y)
        no_improvement_count = 0  # Счётчик итераций без улучшений
        epsilon = 1e-4  # Порог для критерия остановки по стабильности

        for iteration in range(self.max_iterations):  # Цикл по итерациям
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)  # Сортировка антител по BG-аффинности
            selected = S_b[:self.n_b]  # Выбор n_b лучших антител

            current_fitness = self.f(S_b[0].x, S_b[0].y) if S_b else 1.0  # Значение f(x, y) лучшего антитела
            current_mutation_rate = self.mutation_rate if current_fitness > 0.01 else 0.1  # Адаптивная мутация

            clones = []  # Пустой список для клонов
            for ab in selected:  # Цикл по выбранным антителам
                clones.extend(self.clone_antibody(ab))  # Добавление клонов антитела
            clones = [self.mutate_antibody(clone, current_mutation_rate) for clone in clones]  # Мутация всех клонов

            clones = sorted(clones, key=lambda ab: ab.bg_affinity, reverse=True)  # Сортировка клонов по BG-аффинности
            n_d = int(self.b_s * len(clones))  # Число сохраняемых клонов (b_s доля)
            new_memory = clones[:n_d]  # Выбор лучших n_d клонов
            new_memory = [ab for ab in new_memory if ab.bg_affinity >= self.b_b]  # Отбор клонов с аффинностью >= b_b
            S_m.extend(new_memory)  # Добавление отобранных клонов в память

            i = 0  # Индекс для перебора памяти
            while i < len(S_m):  # Цикл по антителам в памяти
                j = i + 1  # Индекс для сравнения
                while j < len(S_m):  # Цикл по последующим антителам
                    if self.compute_bb_affinity(S_m[i], S_m[j]) < self.b_r:  # Проверка BB-аффинности
                        del S_m[j]  # Удаление антитела, если расстояние < b_r
                    else:
                        j += 1  # Переход к следующему антителу
                i += 1  # Переход к следующему антителу
            S_b.extend(S_m)  # Добавление памяти в сеть

            i = 0  # Индекс для перебора сети
            while i < len(S_b):  # Цикл по антителам в сети
                j = i + 1  # Индекс для сравнения
                while j < len(S_b):  # Цикл по последующим антителам
                    if self.compute_bb_affinity(S_b[i], S_b[j]) < self.b_r:  # Проверка BB-аффинности
                        del S_b[j]  # Удаление антитела, если расстояние < b_r
                    else:
                        j += 1  # Переход к следующему антителу
                i += 1  # Переход к следующему антителу

            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)  # Сортировка сети по BG-аффинности
            num_replace = int(self.b_n * self.pop_size)  # Число заменяемых антител (b_n доля)
            S_b = S_b[:self.pop_size - num_replace]  # Удаление худших антител
            S_b.extend(self.initialize_population()[:num_replace])  # Добавление случайных антител

            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)[:self.pop_size]  # Ограничение размера сети

            current_best = S_b[0]  # Текущее лучшее антитело
            current_fitness = self.f(current_best.x, current_best.y)  # Значение f(x, y) лучшего антитела
            if best_solution is None or current_fitness < self.f(best_solution.x, best_solution.y):  # Проверка улучшения
                best_solution = current_best  # Обновление лучшего решения
                iterations_log.append(f"Улучшение на итерации {iteration}: x=[{best_solution.x:.6f}, {best_solution.y:.6f}], f(x)={current_fitness:.6f}")  # Лог улучшения
                no_improvement_count = 0  # Сброс счётчика без улучшений
            else:
                iterations_log.append(f"Итерация {iteration}: x=[{best_solution.x:.6f}, {best_solution.y:.6f}], f(x)={current_fitness:.6f}")  # Лог без улучшения
                no_improvement_count += 1  # Увеличение счётчика без улучшений

            if no_improvement_count >= 5:  # Проверка застревания
                iterations_log.append(f"Предупреждение: нет улучшений в течение {no_improvement_count} итераций")  # Лог предупреждения

            fitness_history.append(current_fitness)  # Добавление f(x, y) в историю
            if len(fitness_history) > 10:  # Проверка длины истории
                fitness_history.pop(0)  # Удаление самого старого значения
                max_diff = max(abs(fitness_history[i] - fitness_history[i-1]) for i in range(1, len(fitness_history)))  # Максимальная разница в истории
                if max_diff < epsilon:  # Проверка стабильности
                    stagnation_count += 1  # Увеличение счётчика стабильности
                else:
                    stagnation_count = 0  # Сброс счётчика стабильности
                if stagnation_count >= 10:  # Проверка критерия остановки
                    iterations_log.append(f"Остановка: значение функции стабильно в течение 10 итераций (разница < {epsilon})")  # Лог остановки
                    break  # Прерывание цикла

            if current_fitness < 1e-6:  # Проверка достижения минимума
                iterations_log.append("Достигнут минимум!")  # Лог достижения минимума
                break  # Прерывание цикла

            trajectory.append([best_solution.x, best_solution.y])  # Добавление лучшей точки в траекторию

        final_point = [best_solution.x, best_solution.y]  # Финальная точка [x, y]
        return final_point, trajectory, "Иммунная сеть завершена", iterations_log  # Возврат результатов

    def plot(self, window):  # Метод для визуализации результатов
        final_point, trajectory, message, iterations_log = self.run()  # Запуск алгоритма и получение результатов

        for log in iterations_log:  # Цикл по логам
            window.log_output(log)  # Вывод каждого лога в консоль программы
        window.log_output(f"Финальная точка: x=[{final_point[0]:.6f}, {final_point[1]:.6f}], f(x)={self.f(*final_point):.6f}")  # Вывод финальной точки
        window.log_output(message)  # Вывод сообщения об окончании

        x = np.linspace(self.range_lower, self.range_upper, 100)  # Создание массива x от range_lower до range_upper
        y = np.linspace(self.range_lower, self.range_upper, 100)  # Создание массива y от range_lower до range_upper
        X, Y = np.meshgrid(x, y)  # Создание сетки координат x, y
        Z = np.array([[self.f(x_i, y_i) for x_i, y_i in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])  # Вычисление f(x, y) для сетки

        fig = go.Figure()  # Создание объекта графика Plotly

        fig.add_trace(go.Surface(  # Добавление 3D-поверхности
            x=X, y=Y, z=Z,  # Координаты x, y, z для поверхности
            colorscale='Viridis',  # Цветовая схема
            showscale=True,  # Отображение шкалы цветов
            opacity=0.8  # Прозрачность поверхности
        ))

        traj_x, traj_y = zip(*trajectory)  # Извлечение x, y из траектории
        traj_z = [self.f(x, y) for x, y in trajectory]  # Вычисление f(x, y) для точек траектории
        fig.add_trace(go.Scatter3d(  # Добавление 3D-траектории
            x=traj_x, y=traj_y, z=traj_z,  # Координаты x, y, z траектории
            mode='lines+markers',  # Режим: линия с маркерами
            line=dict(color='red', width=4),  # Красная линия шириной 4
            marker=dict(size=4),  # Маркеры размером 4
            name='Траектория'  # Название траектории
        ))

        fig.add_trace(go.Scatter3d(  # Добавление финальной точки
            x=[final_point[0]], y=[final_point[1]], z=[self.f(*final_point)],  # Координаты x, y, z точки
            mode='markers',  # Режим: только маркер
            marker=dict(size=8, color='blue', symbol='diamond'),  # Синий ромб размером 8
            name='Минимум'  # Название точки
        ))

        fig.update_layout(  # Настройка макета графика
            title="3D Траектория иммунной сети на функции Розенброка",  # Заголовок графика
            scene=dict(  # Настройка 3D-сцены
                xaxis_title="x",  # Название оси x
                yaxis_title="y",  # Название оси y
                zaxis_title="f(x, y)",  # Название оси z
                zaxis=dict(range=[0, np.max(Z)]),  # Диапазон оси z
            ),
            showlegend=True,  # Отображение легенды
            margin=dict(l=0, r=0, b=0, t=30)  # Минимальные отступы
        )

        html_file = "plot.html"  # Имя файла для сохранения графика
        pio.write_html(fig, file=html_file, auto_open=False)  # Сохранение графика в HTML

        for i in reversed(range(window.graph_layout.count())):  # Цикл по виджетам в layout
            widget = window.graph_layout.itemAt(i).widget()  # Получение виджета
            if widget:  # Проверка наличия виджета
                widget.setParent(None)  # Удаление старого виджета

        web_view = QWebEngineView()  # Создание нового веб-виджета
        web_view.load(f"file:///{html_file}")  # Загрузка HTML-графика
        window.graph_layout.addWidget(web_view)  # Добавление виджета в layout