# Импортируем необходимые библиотеки
import numpy as np  # Для работы с числовыми массивами и операциями
import plotly.graph_objects as go  # Для создания графиков
import plotly.io as pio  # Для вывода графиков в HTML
from PySide6.QtWebEngineWidgets import QWebEngineView  # Для отображения графиков в интерфейсе


class ParticleSwarmOptimization:
    def __init__(self, max_iterations=100, swarmsize=50, minvalues=[-5.12, -5.12],
                 maxvalues=[5.12, 5.12], current_velocity_ratio=0.5, local_velocity_ratio=2.0,
                 global_velocity_ratio=2.0):
        """
        Инициализация алгоритма PSO с заданными параметрами.
        :param max_iterations: Максимальное количество итераций для поиска решения.
        :param swarmsize: Размер роя (количество частиц в рое).
        :param minvalues: Минимальные границы для каждой переменной.
        :param maxvalues: Максимальные границы для каждой переменной.
        :param current_velocity_ratio: Коэффициент инерции для обновления скорости.
        :param local_velocity_ratio: Коэффициент ускорения для локального поиска.
        :param global_velocity_ratio: Коэффициент ускорения для глобального поиска.
        """
        self.max_iterations = max_iterations  # Максимальное количество итераций (по умолчанию 100)
        self.swarmsize = swarmsize  # Размер роя частиц (по умолчанию 50)
        self.minvalues = minvalues  # Нижняя граница значений для каждой переменной (по умолчанию [-5.12, -5.12])
        self.maxvalues = maxvalues  # Верхняя граница значений для каждой переменной (по умолчанию [5.12, 5.12])
        self.current_velocity_ratio = current_velocity_ratio  # Коэффициент инерции для обновления скорости (по умолчанию 0.5)
        self.local_velocity_ratio = local_velocity_ratio  # Коэффициент ускорения для локального поиска (по умолчанию 2.0)
        self.global_velocity_ratio = global_velocity_ratio  # Коэффициент ускорения для глобального поиска (по умолчанию 2.0)

        # Проверка условия для локального и глобального коэффициентов
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4, "Сумма local и global коэффициентов должна быть >= 4"

        # Определение целевой функции (функция Розенброка для двумерной задачи)
        self.f = lambda x1, x2: x1 ** 2 + x2 ** 2  # Функция: f(x1, x2) = x1^2 + x2^2
        self.swarm = self._create_swarm()  # Создание роя частиц


    def get_params(self):
        """
        Получить текущие параметры алгоритма PSO.
        :return: Словарь с текущими параметрами алгоритма.
        """
        return {
            "max_iterations": self.max_iterations,
            "swarmsize": self.swarmsize,
            "minvalues": self.minvalues,
            "maxvalues": self.maxvalues,
            "current_velocity_ratio": self.current_velocity_ratio,
            "local_velocity_ratio": self.local_velocity_ratio,
            "global_velocity_ratio": self.global_velocity_ratio
        }

    def set_params(self, params):
        """
        Установить новые параметры для алгоритма PSO.
        :param params: Словарь с новыми параметрами для алгоритма.
        """
        if "max_iterations" in params:
            self.max_iterations = params["max_iterations"]
        if "swarmsize" in params:
            self.swarmsize = params["swarmsize"]
        if "minvalues" in params:
            self.minvalues = params["minvalues"]
        if "maxvalues" in params:
            self.maxvalues = params["maxvalues"]
        if "current_velocity_ratio" in params:
            self.current_velocity_ratio = params["current_velocity_ratio"]
        if "local_velocity_ratio" in params:
            self.local_velocity_ratio = params["local_velocity_ratio"]
        if "global_velocity_ratio" in params:
            self.global_velocity_ratio = params["global_velocity_ratio"]

        # Проверка для суммы коэффициентов
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4, "Сумма local и global коэффициентов должна быть >= 4"
        self.swarm = self._create_swarm()  # Пересоздание роя с новыми параметрами

    def _create_swarm(self):
        """
        Создание роя частиц с случайными начальными позициями и скоростями.
        :return: Рой частиц, глобально лучшее решение и значение функции в нем.
        """
        minvalues = np.array(self.minvalues)  # Преобразование минимальных значений в массив
        maxvalues = np.array(self.maxvalues)  # Преобразование максимальных значений в массив

        class Particle:
            def __init__(self, outer):
                """
                Инициализация каждой частицы в рое.
                :param outer: Внешний объект класса PSO для доступа к целевой функции и параметрам.
                """
                # Начальная позиция и скорость
                self.position = np.random.rand(2) * (maxvalues - minvalues) + minvalues  # Начальная позиция
                self.velocity = np.random.rand(2) * (maxvalues - minvalues) - (
                            maxvalues - minvalues)  # Начальная скорость
                self.best_position = self.position.copy()  # Локальная лучшая позиция
                self.best_value = outer.f(self.position[0], self.position[1])  # Локальное лучшее значение функции

            def update(self, outer, global_best_position, current_velocity_ratio):
                """
                Обновление позиции и скорости каждой частицы.
                :param outer: Внешний объект класса PSO для доступа к целевой функции и параметрам.
                :param global_best_position: Глобально лучшее положение.
                :param current_velocity_ratio: Текущий коэффициент инерции.
                """
                # Случайные числа для локального и глобального поиска
                rnd_local = np.random.rand(2)  # Случайные значения для локальной части
                rnd_global = np.random.rand(2)  # Случайные значения для глобальной части

                # Сумма коэффициентов ускорения для локального и глобального поиска
                velo_ratio = outer.local_velocity_ratio + outer.global_velocity_ratio

                # Модифицированная формула обновления скорости с коэффициентом common_ratio:
                # common_ratio влияет на величину обновления и адаптируется в зависимости от инерции
                common_ratio = 2.0 * current_velocity_ratio / abs(
                    2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))

                # Обновление скорости частицы на основе локальной и глобальной информации
                new_velocity = (common_ratio * self.velocity +
                                common_ratio * outer.local_velocity_ratio * rnd_local * (
                                        self.best_position - self.position) +  # Локальный поиск
                                common_ratio * outer.global_velocity_ratio * rnd_global * (
                                        global_best_position - self.position))  # Глобальный поиск
                self.velocity = new_velocity  # Обновление скорости
                self.position += self.velocity  # Обновление позиции

                # Вычисление значения целевой функции для новой позиции
                value = outer.f(self.position[0], self.position[1])

                # Обновление локальной лучшей позиции, если новое значение лучше
                if value < self.best_value:
                    self.best_value = value  # Обновление локального лучшего значения
                    self.best_position = self.position.copy()  # Обновление локальной лучшей позиции

        # Создание роя частиц
        swarm = [Particle(self) for _ in range(self.swarmsize)]

        # Нахождение глобального лучшего значения и позиции
        global_best_value = min(p.best_value for p in swarm)
        global_best_position = next(p.best_position for p in swarm if p.best_value == global_best_value)

        return swarm, global_best_position, global_best_value

    def plot(self, window):
        """
        Запуск PSO и отображение результатов в виде 3D-графика.
        :param window: Объект интерфейса для отображения графиков.
        """
        # Инициализация роя частиц и глобального лучшего решения
        swarm, global_best_position, global_best_value = self.swarm
        window.log_output("PSO запущен...")

        # Сохранение траектории лучшей частицы
        trajectory = [global_best_position.copy()]
        tolerance = 1e-6  # Порог для остановки

        # Запуск алгоритма на несколько итераций
        for i in range(self.max_iterations):
            # Затухание инерции: коэффициент инерции уменьшается по мере итераций
            current_velocity_ratio = self.current_velocity_ratio * (1 - i / self.max_iterations)

            # Обновление каждой частицы в рое
            for particle in swarm:
                particle.update(self, global_best_position, current_velocity_ratio)

                # Обновление глобального лучшего решения, если найдено лучшее значение
                if particle.best_value < global_best_value:
                    global_best_value = particle.best_value
                    global_best_position = particle.best_position.copy()

            # Сохранение траектории
            trajectory.append(global_best_position.copy())
            f_val = self.f(global_best_position[0], global_best_position[1])  # Вычисление значения целевой функции
            window.log_output(f"Итерация {i}: x={global_best_position.tolist()}, f(x)={f_val:.6f}")

            # Критерий остановки
            if f_val < tolerance:
                window.log_output(f"Достигнута точность {tolerance}. Остановка на итерации {i}.")
                break

        # Выводим итоговое оптимальное решение
        window.log_output(
            f"Оптимальное решение: x=[{global_best_position[0]:.6f}, {global_best_position[1]:.6f}], f(x)={global_best_value:.6f}")
        window.log_output("PSO завершён")

        # Визуализация результатов
        self._plot_3d_interface(window, trajectory, global_best_position)

    def _plot_3d_interface(self, window, trajectory, solution):
        """
        Рисует 3D-график с поверхностью функции и траекторией.
        :param window: Объект интерфейса для отображения графиков.
        :param trajectory: Траектория движения лучшей частицы.
        :param solution: Оптимальное решение, найденное алгоритмом.
        """
        # Создаём сетку для поверхности
        x1_vals = np.linspace(self.minvalues[0], self.maxvalues[0], 50)
        x2_vals = np.linspace(self.minvalues[1], self.maxvalues[1], 50)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)  # Сетка для отображения поверхности
        Z = np.array(
            [[self.f(x1, x2) for x1, x2 in zip(x1_row, x2_row)] for x1_row, x2_row in zip(X1, X2)])  # Значения на сетке

        # Создаём график
        fig = go.Figure()

        # Поверхность целевой функции
        fig.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='viridis', name="f(x1, x2)"))

        # Траектория лучшей частицы
        trajectory = np.array(trajectory)
        fig.add_trace(go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=[self.f(x, y) for x, y in trajectory],
            mode='lines+markers',
            line=dict(color='blue', width=2),
            marker=dict(size=3, color='blue'),
            name="Траектория"
        ))

        # Точка оптимального решения
        fig.add_trace(go.Scatter3d(
            x=[solution[0]], y=[solution[1]], z=[self.f(solution[0], solution[1])],
            mode='markers',
            marker=dict(size=6, color='red'),
            name="Оптимум"
        ))

        # Настройка графика
        fig.update_layout(
            title="PSO: Траектория и оптимум",
            scene=dict(
                xaxis_title="x1",
                yaxis_title="x2",
                zaxis_title="f(x)",
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        # Сохраняем график в HTML
        html_file = "plot.html"
        pio.write_html(fig, file=html_file, auto_open=False)

        # Очищаем старый график
        for i in reversed(range(window.graph_layout.count())):
            widget = window.graph_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # Встраиваем новый график
        web_view = QWebEngineView()
        web_view.load(f"file:///{html_file}")
        window.graph_layout.addWidget(web_view)
