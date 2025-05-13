import numpy as np  # Импортирует библиотеку numpy для работы с массивами и математическими операциями
import plotly.graph_objects as go  # Импортирует модуль plotly для создания 3D-графиков
import plotly.io as pio  # Импортирует модуль plotly для сохранения графиков в HTML
from PySide6.QtWidgets import QVBoxLayout, QWidget  # Импортирует виджеты PySide6 для интерфейса
from PySide6.QtWebEngineWidgets import QWebEngineView  # Импортирует виджет для отображения HTML-графиков

class BacterialForagingOptimization:  # Определяет класс для алгоритма бактериального поиска
    def __init__(self):  # Инициализирует объект алгоритма
        self.initial_point = [0.5, 0.5]  # Задаёт начальную точку (не используется напрямую, для интерфейса)
        self.max_iterations = 100  # Задаёт максимальное число итераций (не используется, для совместимости)
        self.num_bacteria = 50  # Задаёт число бактерий в популяции
        self.chem_steps = 100  # Задаёт число шагов хемотаксиса в цикле репродукции
        self.repro_steps = 4  # Задаёт число циклов репродукции в цикле ликвидации
        self.elim_steps = 2  # Задаёт число циклов ликвидации/рассеивания
        self.step_size = 0.1  # Задаёт начальную величину шага для хемотаксиса
        self.elim_prob = 0.25  # Задаёт вероятность ликвидации бактерии
        self.elim_count = 10  # Задаёт число бактерий, проверяемых на ликвидацию
        self.bounds_lower = -5  # Задаёт нижнюю границу пространства поиска
        self.bounds_upper = 5  # Задаёт верхнюю границу пространства поиска
        # self.f = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2  # Задаёт функцию Розенброка для минимизации
        self.f = lambda x, y: (x**2 + y**2)

    class Bacterium:  # Определяет класс для представления одной бактерии
        def __init__(self, outer):  # Инициализирует бактерию
            self.dim = 2  # Задаёт размерность пространства (2D: x, y)
            self.bounds = [outer.bounds_lower, outer.bounds_upper]  # Задаёт границы поиска из внешнего объекта
            self.position = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)  # Инициализирует случайную позицию
            self.health = 0  # Задаёт начальное здоровье бактерии (накапливает значения функции)

        def update_health(self, fitness):  # Обновляет здоровье бактерии
            self.health += fitness  # Добавляет значение функции к здоровью (меньше — лучше)

    def get_params(self):  # Возвращает параметры для интерфейса
        return {  # Возвращает словарь с текущими значениями параметров
            "initial_point": self.initial_point,  # Начальная точка
            "max_iterations": self.max_iterations,  # Максимальное число итераций
            "num_bacteria": self.num_bacteria,  # Число бактерий
            "chem_steps": self.chem_steps,  # Число шагов хемотаксиса
            "repro_steps": self.repro_steps,  # Число циклов репродукции
            "elim_steps": self.elim_steps,  # Число циклов ликвидации
            "step_size": self.step_size,  # Начальная величина шага
            "elim_prob": self.elim_prob,  # Вероятность ликвидации
            "elim_count": self.elim_count,  # Число бактерий для ликвидации
            "bounds_lower": self.bounds_lower,  # Нижняя граница
            "bounds_upper": self.bounds_upper  # Верхняя граница
        }

    def set_params(self, params):  # Устанавливает параметры из интерфейса
        if "initial_point" in params:  # Проверяет наличие initial_point
            self.initial_point = params["initial_point"]  # Обновляет начальную точку
        if "max_iterations" in params:  # Проверяет наличие max_iterations
            self.max_iterations = params["max_iterations"]  # Обновляет максимальное число итераций
        if "num_bacteria" in params:  # Проверяет наличие num_bacteria
            self.num_bacteria = params["num_bacteria"]  # Обновляет число бактерий
        if "chem_steps" in params:  # Проверяет наличие chem_steps
            self.chem_steps = params["chem_steps"]  # Обновляет число шагов хемотаксиса
        if "repro_steps" in params:  # Проверяет наличие repro_steps
            self.repro_steps = params["repro_steps"]  # Обновляет число циклов репродукции
        if "elim_steps" in params:  # Проверяет наличие elim_steps
            self.elim_steps = params["elim_steps"]  # Обновляет число циклов ликвидации
        if "step_size" in params:  # Проверяет наличие step_size
            self.step_size = params["step_size"]  # Обновляет величину шага
        if "elim_prob" in params:  # Проверяет наличие elim_prob
            self.elim_prob = params["elim_prob"]  # Обновляет вероятность ликвидации
        if "elim_count" in params:  # Проверяет наличие elim_count
            self.elim_count = params["elim_count"]  # Обновляет число бактерий для ликвидации
        if "bounds_lower" in params:  # Проверяет наличие bounds_lower
            self.bounds_lower = params["bounds_lower"]  # Обновляет нижнюю границу
        if "bounds_upper" in params:  # Проверяет наличие bounds_upper
            self.bounds_upper = params["bounds_upper"]  # Обновляет верхнюю границу

    def run(self):  # Выполняет алгоритм бактериального поиска
        #1 ИНИЦИАЛИЗАЦИЯ
        bacteria = [self.Bacterium(self) for _ in range(self.num_bacteria)]  # Создаёт популяцию бактерий
        best_fitness = float('inf')  # Инициализирует лучшее значение функции как бесконечность
        best_position = None  # Инициализирует лучшую позицию как None
        trajectory = []  # Создаёт список для хранения траектории поиска
        iterations_log = []  # Создаёт список для хранения логов итераций

        for l in range(self.elim_steps):  # Цикл по числу циклов ЛИКВИДАЦИИ
            for r in range(self.repro_steps):  # Цикл по числу циклов РЕПРОДУКЦИИ
                #2 ХЕМОТАКСИС
                for t in range(self.chem_steps):  # Цикл по числу шагов ХЕМОТАКСИСА
                    current_step_size = self.step_size / (t + 1)  # УМЕНЬШАЕТ ШАГ ХЕМОТАКСИСА
                    for bacterium in bacteria:  # Перебирает все бактерии
                        current_fitness = self.f(bacterium.position[0], bacterium.position[1])  # Вычисляет текущее значение функции
                        bacterium.update_health(current_fitness)  # Обновляет здоровье бактерии
                        direction = np.random.uniform(-1, 1, 2)  # Генерирует случайное направление (2D-вектор)
                        direction = direction / np.linalg.norm(direction)  # Нормирует направление (длина = 1)
                        new_position = bacterium.position + current_step_size * direction  # Вычисляет новую позицию
                        new_position = np.clip(new_position, self.bounds_lower, self.bounds_upper)  # Ограничивает позицию границами
                        new_fitness = self.f(new_position[0], new_position[1])  # Вычисляет значение функции в новой позиции
                        if new_fitness < current_fitness:  # Если новая позиция лучше (минимизация)
                            bacterium.position = new_position  # Обновляет позицию бактерии
                            for _ in range(2):  # Выполняет до 2 шагов "плавания"
                                new_position = bacterium.position + current_step_size * direction  # Продолжает движение в том же направлении
                                new_position = np.clip(new_position, self.bounds_lower, self.bounds_upper)  # Ограничивает новую позицию
                                if self.f(new_position[0], new_position[1]) >= new_fitness:  # Если улучшение прекратилось
                                    break  # Прерывает плавание
                                bacterium.position = new_position  # Обновляет позицию
                                new_fitness = self.f(new_position[0], new_position[1])  # Обновляет значение функции
                        else:  # Если улучшения нет, выполняется кувырок
                            direction = np.random.uniform(-1, 1, 2)  # Генерирует новое случайное направление
                            direction = direction / np.linalg.norm(direction)  # Нормирует новое направление
                            bacterium.position = bacterium.position + current_step_size * direction  # Делает шаг в новом направлении
                            bacterium.position = np.clip(bacterium.position, self.bounds_lower, self.bounds_upper)  # Ограничивает позицию

                #3 РЕПРОДУКЦИЯ
                bacteria.sort(key=lambda b: b.health)  # Сортирует бактерии по здоровью (меньше — лучше)
                survivors = bacteria[:self.num_bacteria // 2]  # Выбирает половину лучших бактерий
                bacteria = survivors + [self.Bacterium(self) for _ in range(self.num_bacteria // 2)]  # Создаёт новую популяцию
                for i in range(self.num_bacteria // 2):  # Перебирает индексы для второй половины
                    bacteria[self.num_bacteria // 2 + i].position = survivors[i].position.copy()  # Копирует позиции лучших бактерий
                #4 ОБНОВЛЕНИЕ ЛУЧШЕГО РЕШЕНИЯ
                current_best = min(bacteria, key=lambda b: self.f(b.position[0], b.position[1]))  # Находит лучшую бактерию
                current_fitness = self.f(current_best.position[0], current_best.position[1])  # Вычисляет её значение функции
                if current_fitness < best_fitness:  # Если текущее значение лучше предыдущего лучшего
                    best_fitness = current_fitness  # Обновляет лучшее значение
                    best_position = current_best.position.copy()  # Обновляет лучшую позицию
                trajectory.append(best_position.copy())  # Добавляет лучшую позицию в траекторию
                iterations_log.append(f"Итерация ликвидации {l}, репродукция {r}: x=[{best_position[0]:.6f}, {best_position[1]:.6f}], f(x)={best_fitness:.6f}")  # Логирует итерацию

            #5 ЛИКВИДАЦИЯ/РАССЕИВАНИЕ
            elim_indices = np.random.choice(self.num_bacteria, self.elim_count, replace=False)  # Выбирает индексы для ликвидации
            for i in elim_indices:  # Перебирает выбранные индексы
                if np.random.random() < self.elim_prob:  # Проверяет, ликвидировать ли бактерию
                    bacteria[i] = self.Bacterium(self)  # Заменяет бактерию новой
        #6 ЗАВЕРШЕНИЕ
        return best_position, trajectory, "Бактериальный поиск завершён", iterations_log  # Возвращает результаты

    def plot(self, window):  # Выполняет алгоритм и визуализирует результаты
        final_point, trajectory, message, iterations_log = self.run()  # Выполняет алгоритм и получает результаты
        for log in iterations_log:  # Перебирает логи
            window.log_output(log)  # Выводит каждый лог в консоль интерфейса
        window.log_output(f"Финальная точка: x=[{final_point[0]:.6f}, {final_point[1]:.6f}], f(x)={self.f(*final_point):.6f}")  # Выводит финальную точку
        window.log_output(message)  # Выводит сообщение о завершении
        x = np.linspace(self.bounds_lower, self.bounds_upper, 50)  # Создаёт массив координат x
        y = np.linspace(self.bounds_lower, self.bounds_upper, 50)  # Создаёт массив координат y
        X, Y = np.meshgrid(x, y)  # Создаёт двумерную сетку координат
        Z = np.array([[self.f(x_i, y_i) for x_i, y_i in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])  # Вычисляет значения функции
        fig = go.Figure()  # Создаёт объект фигуры Plotly
        fig.add_trace(go.Surface(  # Добавляет 3D-поверхность функции
            z=Z, x=X, y=Y,  # Указывает данные для поверхности
            colorscale='Viridis',  # Задаёт цветовую схему
            showscale=True  # Показывает цветовую шкалу
        ))
        traj_x, traj_y = zip(*trajectory)  # Извлекает x и y координаты траектории
        traj_z = [self.f(x_i, y_i) for x_i, y_i in zip(traj_x, traj_y)]  # Вычисляет значения функции для траектории
        fig.add_trace(go.Scatter3d(  # Добавляет траекторию поиска
            x=traj_x, y=traj_y, z=traj_z,  # Указывает координаты траектории
            mode='lines+markers',  # Режим: линия с маркерами
            line=dict(color='red', width=4),  # Красная линия толщиной 4
            marker=dict(size=4),  # Маркеры размером 4
            name='Траектория'  # Название траектории
        ))
        fig.add_trace(go.Scatter3d(  # Добавляет финальную точку
            x=[final_point[0]], y=[final_point[1]], z=[self.f(*final_point)],  # Координаты финальной точки
            mode='markers',  # Режим: только маркер
            marker=dict(size=10, color='blue', symbol='diamond'),  # Синий ромб размером 10
            name='Минимум'  # Название точки
        ))
        fig.update_layout(  # Настраивает макет графика
            title="Траектория бактериального поиска",  # Заголовок графика
            scene=dict(  # Настройки 3D-сцены
                xaxis_title="x",  # Подпись оси x
                yaxis_title="y",  # Подпись оси y
                zaxis_title="f(x, y)",  # Подпись оси z
                camera=dict(  # Настройки камеры
                    up=dict(x=0, y=0, z=1),  # Вектор "вверх"
                    center=dict(x=0, y=0, z=0),  # Центр сцены
                    eye=dict(x=1.5, y=1.5, z=1.5)  # Положение камеры
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30)  # Минимальные отступы, заголовок сверху
        )
        html_file = "plot.html"  # Задаёт имя HTML-файла для графика
        pio.write_html(fig, file=html_file, auto_open=False)  # Сохраняет график в HTML
        for i in reversed(range(window.graph_layout.count())):  # Перебирает виджеты в graph_layout
            widget = window.graph_layout.itemAt(i).widget()  # Получает виджет по индексу
            if widget:  # Если виджет существует
                widget.setParent(None)  # Удаляет виджет из интерфейса
        web_view = QWebEngineView()  # Создаёт новый виджет для отображения HTML
        web_view.load(f"file:///{html_file}")  # Загружает HTML-файл с графиком
        window.graph_layout.addWidget(web_view)  # Добавляет виджет в интерфейс