import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PySide6.QtWebEngineWidgets import QWebEngineView

class GeneticAlgorithm:
    def __init__(self, population_size=200, max_iterations=100, mutation_rate=0.1, genetic_bounds=(0, 2), convergence_threshold=1e-6, max_stable_iterations=5):
        self.population_size = int(population_size)  # Преобразуем в целое число
        self.max_iterations = int(max_iterations)  # Преобразуем в целое число
        self.mutation_rate = mutation_rate
        self.genetic_bounds = genetic_bounds  # Используем genetic_bounds вместо bounds
        self.convergence_threshold = convergence_threshold  # Порог для сходимости
        self.max_stable_iterations = max_stable_iterations  # Максимальное количество стабильных итераций

    def get_params(self):
        """Возвращает текущие параметры алгоритма"""
        return {
            "population_size": self.population_size,
            "max_iterations": self.max_iterations,
            "mutation_rate": self.mutation_rate,
            "genetic_bounds": self.genetic_bounds,  # Используем genetic_bounds
            "convergence_threshold": self.convergence_threshold,
            "max_stable_iterations": self.max_stable_iterations
        }

    def set_params(self, params):
        """Устанавливает параметры алгоритма из словаря"""
        if "population_size" in params:
            self.population_size = int(params["population_size"])  # Преобразуем в целое число
        if "max_iterations" in params:
            self.max_iterations = int(params["max_iterations"])  # Преобразуем в целое число
        if "mutation_rate" in params:
            self.mutation_rate = params["mutation_rate"]
        if "genetic_bounds" in params:
            self.genetic_bounds = params["genetic_bounds"]  # Используем genetic_bounds
        if "convergence_threshold" in params:
            self.convergence_threshold = params["convergence_threshold"]
        if "max_stable_iterations" in params:
            self.max_stable_iterations = params["max_stable_iterations"]

    def f(self, x, y):
        """Функция Розенброка"""
        return (1 - x)**2 + 100 * (y - x**2)**2

    def plot(self, window):
        """Точка входа в алгоритм, отрисовка графика и логирование шагов"""
        lower_bound_x, upper_bound_x = self.genetic_bounds  # Границы для x и y

        #1) ИНИЦИАЛИЗАЦИЯ: Генерация начальной популяции с учетом bounds
        population = np.random.uniform(
            low=lower_bound_x,  # нижний предел для x
            high=upper_bound_x,  # верхний предел для x
            size=(self.population_size, 2)
        )

        best_individual = None  # лучшее найденное решение
        best_fitness = float('inf')
        history = []
        stable_iterations = 0  # Количество стабильных итераций

        for generation in range(self.max_iterations):
            #2) Оценка пригодности fitness
            fitness = np.array([self.f(ind[0], ind[1]) for ind in population])  # оценка популяции
            current_best_idx = np.argmin(fitness)  # индекс лучшей особи

            # Обновление лучшего решения
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx].copy()
                history.append((generation, best_individual, best_fitness))
                stable_iterations = 0  # Сбросим счётчик стабильных итераций
            else:
                stable_iterations += 1

            # Печать сообщения о сработавшей точке останова
            if stable_iterations >= self.max_stable_iterations:
                window.log_output(f"Точка останова: Значение функции стабилизировалось за {stable_iterations} итераций.Оптимальное решение: f(x) = {best_fitness}")
                break  # Прерывание, если функция стабилизировалась

            # 3) Селекция (отбор) кандидатов лучшего решения
            selected_indices = []
            for _ in range(self.population_size):
                candidates = np.random.choice(self.population_size, size=5, replace=False)
                winner = candidates[np.argmin(fitness[candidates])]
                selected_indices.append(winner)
            selected_population = population[selected_indices]

            #4) Кроссовер (Скрещивание)
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    parent1, parent2 = selected_population[i], selected_population[i + 1]
                    alpha = np.random.rand()
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                    selected_population[i], selected_population[i + 1] = child1, child2

            # 5) Мутация
            mutation_strength = 0.1 * (0.99 ** generation) #Сила мутации уменьшается с каждым поколением
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    selected_population[i] += np.random.normal(0, mutation_strength, 2)
                    selected_population[i] = np.clip(selected_population[i], self.genetic_bounds[0], self.genetic_bounds[1])

            # 6) Замена худшей особи на лучшую
            worst_idx = np.argmax([self.f(ind[0], ind[1]) for ind in selected_population])
            selected_population[worst_idx] = best_individual

            population = selected_population

            # Логирование прогресса
            window.log_output(f"Итерация {generation}: x = {best_individual}, f(x) = {best_fitness}")

            # Проверка точки останова по значению функции
            if abs(best_fitness) < self.convergence_threshold:
                window.log_output(f"Точка останова: Значение функции стало слишком маленьким (оптимальное решение: f(x) = {best_fitness})")
                break  # Прерывание, если значение функции стало слишком маленьким

        # Отрисовка траектории
        trajectory = [ind for _, ind, _ in history]
        self.plot_trajectory(window, trajectory, best_individual)

    def plot_trajectory(self, window, trajectory, best_individual):
        """Функция для отрисовки траектории на графике с использованием plotly"""
        x_vals = [ind[0] for ind in trajectory]
        y_vals = [ind[1] for ind in trajectory]
        z_vals = [self.f(x, y) for x, y in trajectory]

        # 3D-график через Plotly
        fig = go.Figure()

        # Функция Розенброка в 3D
        x1_vals = np.linspace(self.genetic_bounds[0], self.genetic_bounds[1], 50)
        x2_vals = np.linspace(self.genetic_bounds[0], self.genetic_bounds[1], 50)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = np.array([[self.f(x1, x2) for x1, x2 in zip(x1_row, x2_row)] for x1_row, x2_row in zip(X1, X2)])

        fig.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='plasma', opacity=0.7))

        # Траектория поиска решения
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            marker=dict(size=4, color='blue'),
            name="Траектория"
        ))

        # Получаем последнее значение функции
        last_value = z_vals[-1]  # Это последнее значение функции в последней итерации

        # Форматируем значение с точностью до 6 знаков
        last_value_str = "{:.6f}".format(last_value)  # Форматируем число с точностью до 6 знаков

        # Если значение функции слишком мало, выводим 0
        if abs(last_value) < 1e-6:
            last_value_str = "0.000000"  # Если значение слишком маленькое, отображаем его как 0

        # Отображение оптимума на графике
        fig.add_trace(go.Scatter3d(
            x=[best_individual[0]], y=[best_individual[1]], z=[last_value],
            mode='markers',
            marker=dict(size=6, color='red'),
            name=f"Оптимум (f(x) = {last_value_str})"  # Используем последнее значение функции
        ))

        # Обновление макета
        fig.update_layout(
            title="Генетический алгоритм для функции Розенброка",
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




