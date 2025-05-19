import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PySide6.QtWidgets import QVBoxLayout, QWidget
from PySide6.QtWebEngineWidgets import QWebEngineView
import time

class BacterialForagingOptimization:
    def __init__(self):
        self.initial_point = [0.5, 0.5]
        self.max_iterations = 100
        self.num_bacteria = 50
        self.chem_steps = 100
        self.repro_steps = 4
        self.elim_steps = 2
        self.step_size = 0.1
        self.elim_prob = 0.25
        self.elim_count = 10
        self.bounds_lower = -5
        self.bounds_upper = 5
        self.dimension = 2
        self.f = lambda x, y: x**2 + y**2

    class Bacterium:
        def __init__(self, outer):
            self.dim = outer.dimension
            self.bounds = [outer.bounds_lower, outer.bounds_upper]
            self.position = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
            self.health = 0

        def update_health(self, fitness):
            self.health += fitness

    def get_params(self):
        return {
            "initial_point": self.initial_point,
            "max_iterations": self.max_iterations,
            "num_bacteria": self.num_bacteria,
            "chem_steps": self.chem_steps,
            "repro_steps": self.repro_steps,
            "elim_steps": self.elim_steps,
            "step_size": self.step_size,
            "elim_prob": self.elim_prob,
            "elim_count": self.elim_count,
            "bounds_lower": self.bounds_lower,
            "bounds_upper": self.bounds_upper,
            "dimension": self.dimension
        }

    def set_params(self, params):
        if "initial_point" in params:
            self.initial_point = params["initial_point"]
        if "max_iterations" in params:
            self.max_iterations = params["max_iterations"]
        if "num_bacteria" in params:
            self.num_bacteria = params["num_bacteria"]
        if "chem_steps" in params:
            self.chem_steps = params["chem_steps"]
        if "repro_steps" in params:
            self.repro_steps = params["repro_steps"]
        if "elim_steps" in params:
            self.elim_steps = params["elim_steps"]
        if "step_size" in params:
            self.step_size = params["step_size"]
        if "elim_prob" in params:
            self.elim_prob = params["elim_prob"]
        if "elim_count" in params:
            self.elim_count = params["elim_count"]
        if "bounds_lower" in params:
            self.bounds_lower = params["bounds_lower"]
        if "bounds_upper" in params:
            self.bounds_upper = params["bounds_upper"]
        if "dimension" in params:
            self.dimension = params["dimension"]

    def run(self):
        bacteria = [self.Bacterium(self) for _ in range(self.num_bacteria)]
        best_fitness = float('inf')
        best_position = None
        trajectory = []
        iterations_log = []
        times = []
        start_time = time.time()

        for l in range(self.elim_steps):
            for r in range(self.repro_steps):
                for t in range(self.chem_steps):
                    current_step_size = self.step_size / (t + 1)
                    for bacterium in bacteria:
                        current_fitness = self.f(*bacterium.position)
                        bacterium.update_health(current_fitness)
                        direction = np.random.uniform(-1, 1, self.dimension)
                        direction = direction / np.linalg.norm(direction)
                        new_position = bacterium.position + current_step_size * direction
                        new_position = np.clip(new_position, self.bounds_lower, self.bounds_upper)
                        new_fitness = self.f(*new_position)
                        if new_fitness < current_fitness:
                            bacterium.position = new_position
                            for _ in range(2):
                                new_position = bacterium.position + current_step_size * direction
                                new_position = np.clip(new_position, self.bounds_lower, self.bounds_upper)
                                if self.f(*new_position) >= new_fitness:
                                    break
                                bacterium.position = new_position
                                new_fitness = self.f(*new_position)
                        else:
                            direction = np.random.uniform(-1, 1, self.dimension)
                            direction = direction / np.linalg.norm(direction)
                            bacterium.position = bacterium.position + current_step_size * direction
                            bacterium.position = np.clip(bacterium.position, self.bounds_lower, self.bounds_upper)

                bacteria.sort(key=lambda b: b.health)
                survivors = bacteria[:self.num_bacteria // 2]
                bacteria = survivors + [self.Bacterium(self) for _ in range(self.num_bacteria // 2)]
                for i in range(self.num_bacteria // 2):
                    bacteria[self.num_bacteria // 2 + i].position = survivors[i].position.copy()
                current_best = min(bacteria, key=lambda b: self.f(*b.position))
                current_fitness = self.f(*current_best.position)
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_position = current_best.position.copy()
                trajectory.append(best_position.copy())
                iterations_log.append(f"Итерация ликвидации {l}, репродукция {r}: x=[{best_position[0]:.6f}, {best_position[1]:.6f}], f(x)={best_fitness:.6f}")
                times.append((l * self.repro_steps + r + 1, time.time() - start_time))

            elim_indices = np.random.choice(self.num_bacteria, self.elim_count, replace=False)
            for i in elim_indices:
                if np.random.random() < self.elim_prob:
                    bacteria[i] = self.Bacterium(self)

        fitness = np.array([self.f(*b.position) for b in bacteria])
        top_indices = np.argsort(fitness)[:10]
        top_positions = [bacteria[i].position.copy() for i in top_indices]

        return best_position, trajectory, "Бактериальный поиск завершён", iterations_log, top_positions, times, best_fitness

    def plot(self, window):
        final_point, trajectory, message, iterations_log, top_positions, times, final_value = self.run()
        for log in iterations_log:
            window.log_output(log)
        window.log_output(f"Финальная точка: x=[{final_point[0]:.6f}, {final_point[1]:.6f}], f(x)={final_value:.6f}")
        window.log_output(message)
        x = np.linspace(self.bounds_lower, self.bounds_upper, 50)
        y = np.linspace(self.bounds_lower, self.bounds_upper, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[self.f(x_i, y_i) for x_i, y_i in zip(x_row, y_row)] for x_row, y_row in zip(X, Y)])
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', showscale=True))
        traj_x, traj_y = zip(*trajectory)
        traj_z = [self.f(x_i, y_i) for x_i, y_i in zip(traj_x, traj_y)]
        fig.add_trace(go.Scatter3d(x=traj_x, y=traj_y, z=traj_z, mode='lines+markers', line=dict(color='red', width=4), marker=dict(size=4), name='Траектория'))
        fig.add_trace(go.Scatter3d(x=[final_point[0]], y=[final_point[1]], z=[self.f(*final_point)], mode='markers', marker=dict(size=10, color='blue', symbol='diamond'), name='Минимум'))
        fig.update_layout(title="Траектория бактериального поиска", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)", camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.5))), margin=dict(l=0, r=0, b=0, t=30))
        html_file = "plot.html"
        pio.write_html(fig, file=html_file, auto_open=False)
        for i in reversed(range(window.graph_layout.count())):
            widget = window.graph_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        web_view = QWebEngineView()
        web_view.load(f"file:///{html_file}")
        window.graph_layout.addWidget(web_view)
        return times, final_value