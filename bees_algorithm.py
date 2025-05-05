import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PySide6.QtWebEngineWidgets import QWebEngineView
import random

class BeesAlgorithm:
    def __init__(self, scoutbeecount=100, selectedbeecount=5, bestbeecount=20, selsitescount=10,
                 bestsitescount=3, range_lower=-2.0, range_upper=2.0, range_shrink=0.95,
                 max_iterations=30, max_stagnation=5, convergence_threshold=1e-8):
        self.scoutbeecount = int(scoutbeecount)
        self.selectedbeecount = int(selectedbeecount)
        self.bestbeecount = int(bestbeecount)
        self.selsitescount = int(selsitescount)
        self.bestsitescount = int(bestsitescount)
        self.range_lower = range_lower
        self.range_upper = range_upper
        self.range_shrink = range_shrink
        self.max_iterations = int(max_iterations)
        self.max_stagnation = int(max_stagnation)
        self.convergence_threshold = convergence_threshold

    def get_params(self):
        """Возвращает текущие параметры алгоритма"""
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
        """Устанавливает параметры алгоритма из словаря"""
        if "scoutbeecount" in params:
            self.scoutbeecount = int(params["scoutbeecount"])
        if "selectedbeecount" in params:
            self.selectedbeecount = int(params["selectedbeecount"])
        if "bestbeecount" in params:
            self.bestbeecount = int(params["bestbeecount"])
        if "selsitescount" in params:
            self.selsitescount = int(params["selsitescount"])
        if "bestsitescount" in params:
            self.bestsitescount = int(params["bestsitescount"])
        if "range_lower" in params:
            self.range_lower = params["range_lower"]
        if "range_upper" in params:
            self.range_upper = params["range_upper"]
        if "range_shrink" in params:
            self.range_shrink = params["range_shrink"]
        if "max_iterations" in params:
            self.max_iterations = int(params["max_iterations"])
        if "max_stagnation" in params:
            self.max_stagnation = int(params["max_stagnation"])
        if "convergence_threshold" in params:
            self.convergence_threshold = params["convergence_threshold"]

    def f(self, x, y):
        """Функция Розенброка"""
        return (1 - x)**2 + 100 * (y - x**2)**2

    class FloatBee:
        def __init__(self, outer):
            self.minval = [outer.range_lower] * 2
            self.maxval = [outer.range_upper] * 2
            self.position = np.array([random.uniform(self.minval[n], self.maxval[n]) for n in range(2)])
            self.fitness = outer.f(self.position[0], self.position[1])

        def calcfitness(self, outer):
            self.fitness = outer.f(self.position[0], self.position[1])

        def goto(self, otherpos, range_list, outer):
            self.position = np.array([otherpos[n] + random.uniform(-range_list[n], range_list[n])
                                    for n in range(len(otherpos))])
            self.checkposition()
            self.calcfitness(outer)

        def gotorandom(self, outer):
            self.position = np.array([random.uniform(self.minval[n], self.maxval[n])
                                    for n in range(2)])
            self.checkposition()
            self.calcfitness(outer)

        def checkposition(self):
            self.position = np.clip(self.position, self.minval, self.maxval)

        def otherpatch(self, bee_list, range_list):
            if not bee_list:
                return True
            for curr_bee in bee_list:
                position = curr_bee.position
                if any(abs(self.position[n] - position[n]) > range_list[n] for n in range(2)):
                    return True
            return False

    def plot(self, window):
        """Точка входа в алгоритм, отрисовка графика и логирование шагов"""
        # Инициализация улья
        beecount = self.scoutbeecount + self.selectedbeecount * self.selsitescount + self.bestbeecount * self.bestsitescount
        swarm = [self.FloatBee(self) for _ in range(beecount)]
        range_list = [(self.range_upper - self.range_lower) / 4] * 2  # Уменьшен начальный радиус поиска
        best_fitness = float('inf')
        best_position = None
        trajectory = []
        stagnation_counter = 0

        for iteration in range(self.max_iterations):
            # Сортировка пчел по возрастанию fitness
            swarm.sort(key=lambda x: x.fitness)
            current_best_fitness = swarm[0].fitness
            current_best_position = swarm[0].position.copy()

            # Обновление лучшего решения
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_position = current_best_position
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Логирование
            window.log_output(f"Итерация {iteration}: x={best_position}, f(x)={best_fitness}")
            trajectory.append(best_position.copy())

            # Проверка точки останова по значению функции
            if abs(best_fitness) < self.convergence_threshold:
                window.log_output(f"Точка останова: Значение функции стало слишком маленьким (f(x) = {best_fitness})")
                break

            # Проверка точки останова по стагнации
            if stagnation_counter >= self.max_stagnation:
                window.log_output(
                    f"Точка останова: Значение функции стабилизировалось за {stagnation_counter} последних итераций (f(x) = {best_fitness})")
                range_list = [r * self.range_shrink for r in range_list]  # Сужение диапазона при стагнации
                break

            # Выбор элитных и перспективных участков
            bestsites = [swarm[0]]
            curr_index = 1
            while len(bestsites) < self.bestsitescount and curr_index < len(swarm):
                if swarm[curr_index].otherpatch(bestsites, range_list):
                    bestsites.append(swarm[curr_index])
                curr_index += 1

            selsites = []
            while len(selsites) < self.selsitescount and curr_index < len(swarm):
                if (swarm[curr_index].otherpatch(bestsites, range_list) and
                        swarm[curr_index].otherpatch(selsites, range_list)):
                    selsites.append(swarm[curr_index])
                curr_index += 1

            # Отправка рабочих пчел
            bee_index = 1
            for best_bee in bestsites:
                best_range = [r * 0.5 for r in range_list]  # Уменьшенный радиус для элитных участков
                for _ in range(self.bestbeecount):
                    if bee_index >= len(swarm):
                        break
                    if swarm[bee_index] not in bestsites and swarm[bee_index] not in selsites:
                        swarm[bee_index].goto(best_bee.position, best_range, self)
                    bee_index += 1

            for sel_bee in selsites:
                for _ in range(self.selectedbeecount):
                    if bee_index >= len(swarm):
                        break
                    if swarm[bee_index] not in bestsites and swarm[bee_index] not in selsites:
                        swarm[bee_index].goto(sel_bee.position, range_list, self)
                    bee_index += 1

            # Разведчики
            for bee in swarm[bee_index:]:
                bee.gotorandom(self)

        else:
            # Срабатывает, если цикл завершился по достижению max_iterations
            window.log_output(
                f"Точка останова: Достигнуто максимальное количество итераций ({self.max_iterations}) (f(x) = {best_fitness})")

        # Отрисовка траектории
        self.plot_trajectory(window, trajectory, best_position)

    def plot_trajectory(self, window, trajectory, best_individual):
        """Функция для отрисовки траектории на графике с использованием plotly"""
        x_vals = [ind[0] for ind in trajectory]
        y_vals = [ind[1] for ind in trajectory]
        z_vals = [self.f(x, y) for x, y in trajectory]

        # 3D-график через Plotly
        fig = go.Figure()

        # Функция Розенброка в 3D
        x1_vals = np.linspace(self.range_lower, self.range_upper, 50)
        x2_vals = np.linspace(self.range_lower, self.range_upper, 50)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = np.array([[self.f(x1, x2) for x1, x2 in zip(x1_row, x2_row)] for x1_row, x2_row in zip(X1, X2)])

        fig.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='viridis', opacity=0.7))

        # Траектория поиска решения
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            name="Траектория"
        ))

        # Получаем последнее значение функции
        last_value = z_vals[-1]
        last_value_str = "{:.6f}".format(last_value)
        if abs(last_value) < 1e-6:
            last_value_str = "0.000000"

        # Отображение оптимума на графике
        fig.add_trace(go.Scatter3d(
            x=[best_individual[0]], y=[best_individual[1]], z=[last_value],
            mode='markers',
            marker=dict(size=6, color='red'),
            name=f"Оптимум (f(x) = {last_value_str})"
        ))

        # Обновление макета
        fig.update_layout(
            title="Пчелиный алгоритм для функции Розенброка",
            scene=dict(
                xaxis_title="x1",
                yaxis_title="x2",
                zaxis_title="F(x)"
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )

        # Сохранение графика в HTML файл
        html_file = "plot.html"
        pio.write_html(fig, file=html_file, auto_open=False)

        # Загружаем график в QWebEngineView
        for i in reversed(range(window.graph_layout.count())):
            widget = window.graph_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        web_view = QWebEngineView()
        web_view.load(f"file:///{html_file}")
        window.graph_layout.addWidget(web_view)