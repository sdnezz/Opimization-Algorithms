import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtWebEngineWidgets import QWebEngineView

class ImmuneNetworkOptimization:
    def __init__(self):
        self.initial_point = [1.0, 1.0] #стартовая точка
        self.max_iterations = 100
        self.pop_size = 100
        self.n_b = 20 #число лучших антител для клонирования
        self.n_c = 10 #максимальное число клонов для одного антитела
        self.b_s = 0.1 #доля лучших клонов для сохранения
        self.b_b = 0.01 #порог афинности для отбора клонов
        self.b_r = 0.02 #попрог расстояния между антителами
        self.b_n = 0.1 #доля антител, заменяемых случайными
        self.mutation_rate = 0.5 #сила мутации
        self.range_lower = -2 #нижняя граница пространства поиска
        self.range_upper = 2 #верхняя граница пространства поиска
        self.f = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2 #Розенброк

    class Antibody:
        def __init__(self, x, y, outer):
            self.x = x
            self.y = y
            self.bg_affinity = 1 / (1 + outer.f(x, y)) #бг-афинность

    def get_params(self):
        return {
            "initial_point": self.initial_point,
            "max_iterations": self.max_iterations,
            "pop_size": self.pop_size,
            "n_b": self.n_b,
            "n_c": self.n_c,
            "b_s": self.b_s,
            "b_b": self.b_b,
            "b_r": self.b_r,
            "b_n": self.b_n,
            "mutation_rate": self.mutation_rate,
            "range_lower": self.range_lower,
            "range_upper": self.range_upper
        }

    def set_params(self, params):
        if "initial_point" in params:
            self.initial_point = params["initial_point"]
        if "max_iterations" in params:
            self.max_iterations = params["max_iterations"]
        if "pop_size" in params:
            self.pop_size = params["pop_size"]
        if "n_b" in params:
            self.n_b = params["n_b"]
        if "n_c" in params:
            self.n_c = params["n_c"]
        if "b_s" in params:
            self.b_s = params["b_s"]
        if "b_b" in params:
            self.b_b = params["b_b"]
        if "b_r" in params:
            self.b_r = params["b_r"]
        if "b_n" in params:
            self.b_n = params["b_n"]
        if "mutation_rate" in params:
            self.mutation_rate = params["mutation_rate"]
        if "range_lower" in params:
            self.range_lower = params["range_lower"]
        if "range_upper" in params:
            self.range_upper = params["range_upper"]

    def compute_bb_affinity(self, ab1, ab2): #вычисляет ВВ-афинность(евклидово расстояние между телами)
        return np.sqrt((ab1.x - ab2.x)**2 + (ab1.y - ab2.y)**2)

    def initialize_population(self): #Создает начальную популяцию из pop_size случайных антител от range_lower до range_upper
        return [self.Antibody(np.random.uniform(self.range_lower, self.range_upper),
                              np.random.uniform(self.range_lower, self.range_upper), self)
                for _ in range(self.pop_size)]

    def clone_antibody(self, antibody): #создает от 1 до n_с клонов антитела, пропорционально его афинности
        num_clones = int(1 + (self.n_c - 1) * antibody.bg_affinity) #для каждого вызывается clone_antibody
        return [self.Antibody(antibody.x, antibody.y, self) for _ in range(num_clones)]

    def mutate_antibody(self, antibody): #изменяет координаты антитела на случайный шаг, пропорциональный current_mutation_rate, с ограничением в [range_lower, range_upper].
        x_new = antibody.x + self.mutation_rate * np.random.uniform(-0.5, 0.5)
        y_new = antibody.y + self.mutation_rate * np.random.uniform(-0.5, 0.5)
        x_new = np.clip(x_new, self.range_lower, self.range_upper)
        y_new = np.clip(y_new, self.range_lower, self.range_upper)
        return self.Antibody(x_new, y_new, self)

    def run(self): #тут понятно основной цикл
        ###ИНИЦИАЛИЗАЦИЯ
        S_b = self.initialize_population()
        S_m = []
        best_solution = None
        trajectory = []
        iterations_log = []
        fitness_history = []
        stagnation_count = 0
        epsilon = 1e-6
        ###КОНЕЦ ИНИЦИАЛИЗАЦИИ

        ###ОСНОВНОЙ ЦИКЛ
        for iteration in range(self.max_iterations):
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True) #ОТБОР ЛУЧШИХ АНТИТЕЛ
            selected = S_b[:self.n_b] #КОНЕЦ ОТБОРА

            clones = [] #КЛОНИРОВАНИЕ
            for ab in selected:
                clones.extend(self.clone_antibody(ab))
            clones = [self.mutate_antibody(clone) for clone in clones] #Каждый клон мутирует

            clones = sorted(clones, key=lambda ab: ab.bg_affinity, reverse=True) # Отбор лучших клонов
            n_d = int(self.b_s * len(clones)) #СОХРАНЯЕМЫЕ КЛОНЫ
            new_memory = clones[:n_d]
            new_memory = [ab for ab in new_memory if ab.bg_affinity >= self.b_b]
            S_m.extend(new_memory)
            #СЖАТИЕ ПАМЯТИ
            i = 0
            while i < len(S_m):
                j = i + 1
                while j < len(S_m):
                    if self.compute_bb_affinity(S_m[i], S_m[j]) < self.b_r:
                        del S_m[j]
                    else:
                        j += 1
                i += 1
            S_b.extend(S_m)
            #КОНЕЦ СЖАТИЯ ПАМЯТИ

            #СЖАТИЕ СЕТИ
            i = 0
            while i < len(S_b):
                j = i + 1
                while j < len(S_b):
                    if self.compute_bb_affinity(S_b[i], S_b[j]) < self.b_r:
                        del S_b[j]
                    else:
                        j += 1
                i += 1
            #КОНЕЦ СЖАТИЯ СЕТИ

            #ЗАМЕНА ХУДШИХ АНТИТЕЛ
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)
            num_replace = int(self.b_n * self.pop_size) #ЗАМЕНЯЕМЫЕ АНТИТЕЛА
            S_b = S_b[:self.pop_size - num_replace]
            S_b.extend(self.initialize_population()[:num_replace])

            #ОГРАНИЧЕНИЕ ПОПУЛЯЦИИ
            S_b = sorted(S_b, key=lambda ab: ab.bg_affinity, reverse=True)[:self.pop_size]

            #ОБНОВЛЕНИЕ ЛУЧШЕГО РЕШЕНИЯ
            current_best = S_b[0]
            current_fitness = self.f(current_best.x, current_best.y) # АДАПТИВНАЯ МУТАЦИЯ
            if best_solution is None or current_fitness < self.f(best_solution.x, best_solution.y):
                best_solution = current_best
                iterations_log.append(f"Улучшение на итерации {iteration}: x=[{best_solution.x:.6f}, {best_solution.y:.6f}], f(x)={current_fitness:.6f}")
            else:
                iterations_log.append(f"Итерация {iteration}: x=[{best_solution.x:.6f}, {best_solution.y:.6f}], f(x)={current_fitness:.6f}")

            #ПРОВЕРКА КРИТЕРИЯ ОСТАНОВКИ
            fitness_history.append(current_fitness)
            if len(fitness_history) > 10:
                fitness_history.pop(0)
                max_diff = max(abs(fitness_history[i] - fitness_history[i-1]) for i in range(1, len(fitness_history))) #ОСТАНОВКА ПО СТАБИЛЬНОСТИ
                if max_diff < epsilon:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                if stagnation_count >= 10:
                    iterations_log.append(f"Остановка: значение функции стабильно в течение 10 итераций (разница < {epsilon})")
                    break

            if current_fitness < 1e-6: #ОСТАНОВКА ПО ЗНАЧЕНИЮ
                iterations_log.append("Достигнут минимум!")
                break

            trajectory.append([best_solution.x, best_solution.y])

        final_point = [best_solution.x, best_solution.y]
        return final_point, trajectory, "Иммунная сеть завершена", iterations_log

    def plot(self, window): #тут понятно выполняет run(), ну дефолт короче
        final_point, trajectory, message, iterations_log = self.run()

        for log in iterations_log:
            window.log_output(log)
        window.log_output(f"Финальная точка: x=[{final_point[0]:.6f}, {final_point[1]:.6f}], f(x)={self.f(*final_point):.6f}")
        window.log_output(message)

        x = np.linspace(self.range_lower, self.range_upper, 100)
        y = np.linspace(self.range_lower, self.range_upper, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.f(x_i, y_i) for x_i, y_i in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])

        fig = go.Figure()
        fig.add_trace(go.Contour(
            z=Z, x=x, y=y,
            colorscale='Viridis',
            contours=dict(showlabels=True),
            showscale=True
        ))
        traj_x, traj_y = zip(*trajectory)
        fig.add_trace(go.Scatter(
            x=traj_x, y=traj_y,
            mode='lines+markers',
            line=dict(color='red', width=2),
            marker=dict(size=6),
            name='Траектория'
        ))
        fig.add_trace(go.Scatter(
            x=[final_point[0]], y=[final_point[1]],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='star'),
            name='Минимум'
        ))

        fig.update_layout(
            title="Траектория иммунной сети",
            xaxis_title="x",
            yaxis_title="y",
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=30)
        )

        html_file = "plot.html"
        pio.write_html(fig, file=html_file, auto_open=False)

        for i in reversed(range(window.graph_layout.count())):
            widget = window.graph_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        web_view = QWebEngineView()
        web_view.load(f"file:///{html_file}")
        window.graph_layout.addWidget(web_view)