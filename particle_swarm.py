import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PySide6.QtWebEngineWidgets import QWebEngineView
import time

class ParticleSwarmOptimization:
    def __init__(self, max_iterations=100, swarmsize=50, minvalues=[-5, -5],
                 maxvalues=[5, 5], current_velocity_ratio=0.5, local_velocity_ratio=2.0,
                 global_velocity_ratio=2.0, initial_positions=None):
        self.max_iterations = max_iterations
        self.swarmsize = swarmsize
        self.minvalues = minvalues
        self.maxvalues = maxvalues
        self.current_velocity_ratio = current_velocity_ratio
        self.local_velocity_ratio = local_velocity_ratio
        self.global_velocity_ratio = global_velocity_ratio
        self.initial_positions = initial_positions
        self.dimension = 2
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4, "Сумма local и global коэффициентов должна быть >= 4"
        self.f = lambda x1, x2: x1**2 + x2**2
        self.swarm = self._create_swarm()

    def get_params(self):
        return {
            "max_iterations": self.max_iterations,
            "swarmsize": self.swarmsize,
            "minvalues": self.minvalues,
            "maxvalues": self.maxvalues,
            "current_velocity_ratio": self.current_velocity_ratio,
            "local_velocity_ratio": self.local_velocity_ratio,
            "global_velocity_ratio": self.global_velocity_ratio,
            "dimension": self.dimension
        }

    def set_params(self, params):
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
        if "dimension" in params:
            self.dimension = params["dimension"]
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4, "Сумма local и global коэффициентов должна быть >= 4"
        self.swarm = self._create_swarm()

    def _create_swarm(self):
        minvalues = np.array(self.minvalues)
        maxvalues = np.array(self.maxvalues)

        class Particle:
            def __init__(self, outer, position=None):
                if position is not None:
                    self.position = np.array(position)
                else:
                    self.position = np.random.rand(outer.dimension) * (maxvalues - minvalues) + minvalues
                self.velocity = np.random.rand(outer.dimension) * (maxvalues - minvalues) - (maxvalues - minvalues)
                self.best_position = self.position.copy()
                self.best_value = outer.f(*self.position)

            def update(self, outer, global_best_position, current_velocity_ratio):
                rnd_local = np.random.rand(outer.dimension)
                rnd_global = np.random.rand(outer.dimension)
                velo_ratio = outer.local_velocity_ratio + outer.global_velocity_ratio
                common_ratio = 2.0 * current_velocity_ratio / abs(2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))
                new_velocity = (common_ratio * self.velocity +
                                common_ratio * outer.local_velocity_ratio * rnd_local * (self.best_position - self.position) +
                                common_ratio * outer.global_velocity_ratio * rnd_global * (global_best_position - self.position))
                self.velocity = new_velocity
                self.position += self.velocity
                self.position = np.clip(self.position, minvalues, maxvalues)
                value = outer.f(*self.position)
                if value < self.best_value:
                    self.best_value = value
                    self.best_position = self.position.copy()

        swarm = []
        if self.initial_positions is not None:
            for pos in self.initial_positions:
                swarm.append(Particle(self, pos))
            for _ in range(self.swarmsize - len(self.initial_positions)):
                swarm.append(Particle(self))
        else:
            swarm = [Particle(self) for _ in range(self.swarmsize)]

        global_best_value = min(p.best_value for p in swarm)
        global_best_position = next(p.best_position for p in swarm if p.best_value == global_best_value)
        return swarm, global_best_position, global_best_value

    def run(self):
        swarm, global_best_position, global_best_value = self.swarm
        trajectory = [global_best_position.copy()]
        iterations_log = []
        times = []
        start_time = time.time()
        tolerance = 1e-6

        for i in range(self.max_iterations):
            current_velocity_ratio = self.current_velocity_ratio * (1 - i / self.max_iterations)
            for particle in swarm:
                particle.update(self, global_best_position, current_velocity_ratio)
                if particle.best_value < global_best_value:
                    global_best_value = particle.best_value
                    global_best_position = particle.best_position.copy()
            trajectory.append(global_best_position.copy())
            f_val = self.f(*global_best_position)
            iterations_log.append(f"Итерация {i}: x=[{global_best_position[0]:.6f}, {global_best_position[1]:.6f}], f(x)={f_val:.6f}")
            times.append((i + 1, time.time() - start_time))
            if f_val < tolerance:
                iterations_log.append(f"Достигнута точность {tolerance}. Остановка на итерации {i}.")
                break

        return global_best_position, trajectory, "PSO завершён", iterations_log, times, global_best_value

    def plot(self, window):
        final_point, trajectory, message, iterations_log, times, final_value = self.run()
        for log in iterations_log:
            window.log_output(log)
        window.log_output(f"Финальная точка: x=[{final_point[0]:.6f}, {final_point[1]:.6f}], f(x)={final_value:.6f}")
        window.log_output(message)
        x1_vals = np.linspace(self.minvalues[0], self.maxvalues[0], 50)
        x2_vals = np.linspace(self.minvalues[1], self.maxvalues[1], 50)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = np.array([[self.f(x1, x2) for x1, x2 in zip(x1_row, x2_row)] for x1_row, x2_row in zip(X1, X2)])
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='viridis', name="f(x1, x2)"))
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
        fig.add_trace(go.Scatter3d(
            x=[final_point[0]], y=[final_point[1]], z=[self.f(*final_point)],
            mode='markers',
            marker=dict(size=6, color='red'),
            name="Оптимум"
        ))
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