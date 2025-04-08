# particle_swarm.py
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PySide6.QtWebEngineWidgets import QWebEngineView

class ParticleSwarmOptimization:
    def __init__(self, max_iterations=100, swarmsize=50, minvalues=[-5.12, -5.12],
                 maxvalues=[5.12, 5.12], current_velocity_ratio=0.5, local_velocity_ratio=2.0,
                 global_velocity_ratio=2.0):
        self.max_iterations = max_iterations
        self.swarmsize = swarmsize
        self.minvalues = minvalues
        self.maxvalues = maxvalues
        self.current_velocity_ratio = current_velocity_ratio
        self.local_velocity_ratio = local_velocity_ratio
        self.global_velocity_ratio = global_velocity_ratio
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
            "global_velocity_ratio": self.global_velocity_ratio
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
        assert self.local_velocity_ratio + self.global_velocity_ratio >= 4, "Сумма local и global коэффициентов должна быть >= 4"
        self.swarm = self._create_swarm()

    def _create_swarm(self):
        minvalues = np.array(self.minvalues)
        maxvalues = np.array(self.maxvalues)

        class Particle:
            def __init__(self, outer):
                self.position = np.random.rand(2) * (maxvalues - minvalues) + minvalues
                self.velocity = np.random.rand(2) * (maxvalues - minvalues) - (maxvalues - minvalues) 
                self.best_position = self.position.copy()
                self.best_value = outer.f(self.position[0], self.position[1])

            def update(self, outer, global_best_position, current_velocity_ratio):
                rnd_local = np.random.rand(2)
                rnd_global = np.random.rand(2)
                velo_ratio = outer.local_velocity_ratio + outer.global_velocity_ratio
                # тут используется модифицированная формула с параметром common_ratio (я на листочке расписал, если надо)
                common_ratio = 2.0 * current_velocity_ratio / abs(2.0 - velo_ratio - np.sqrt(velo_ratio ** 2 - 4.0 * velo_ratio))
                
                new_velocity = (common_ratio * self.velocity +
                                common_ratio * outer.local_velocity_ratio * rnd_local * (self.best_position - self.position) +
                                common_ratio * outer.global_velocity_ratio * rnd_global * (global_best_position - self.position))
                self.velocity = new_velocity
                self.position += self.velocity
                value = outer.f(self.position[0], self.position[1])
                if value < self.best_value: # если частица нашла лучшее значение, то обновляется ее бест велью и позишн
                    self.best_value = value
                    self.best_position = self.position.copy()

        swarm = [Particle(self) for _ in range(self.swarmsize)]
        global_best_value = min(p.best_value for p in swarm)
        global_best_position = next(p.best_position for p in swarm if p.best_value == global_best_value)
        return swarm, global_best_position, global_best_value

    def plot(self, window):
        """Запускает PSO и рисует 3D-график в интерфейсе"""
        swarm, global_best_position, global_best_value = self.swarm
        window.log_output("PSO запущен...")

        # Сохраняем траекторию лучшей частицы
        trajectory = [global_best_position.copy()]
        tolerance = 1e-6  # Порог для остановки

        for i in range(self.max_iterations):
            # Затухание инерции
            current_velocity_ratio = self.current_velocity_ratio * (1 - i / self.max_iterations) # уменьшается коэффициент инерции (у нас он не константа, а зависит от итерации)
            for particle in swarm:
                particle.update(self, global_best_position, current_velocity_ratio) # тут обновляется скорость и позиция для каждой частицы
                if particle.best_value < global_best_value: # если нашлось бест велью в частице лучшее, чем глобальное бест велью, то обновляется лучшее глобальное значение и позиция 
                    global_best_value = particle.best_value
                    global_best_position = particle.best_position.copy()

            trajectory.append(global_best_position.copy())
            f_val = self.f(global_best_position[0], global_best_position[1])
            window.log_output(f"Итерация {i}: x={global_best_position.tolist()}, f(x)={f_val:.6f}")

            # Критерий остановки 1e-6
            if f_val < tolerance:
                window.log_output(f"Достигнута точность {tolerance}. Остановка на итерации {i}.")
                break

        window.log_output(f"Оптимальное решение: x=[{global_best_position[0]:.6f}, {global_best_position[1]:.6f}], f(x)={global_best_value:.6f}")
        window.log_output("PSO завершён")

        # Визуализация
        self._plot_3d_interface(window, trajectory, global_best_position)

    def _plot_3d_interface(self, window, trajectory, solution):
        """Рисует 3D-график с поверхностью функции и траекторией"""
        # Создаём сетку для поверхности
        x1_vals = np.linspace(self.minvalues[0], self.maxvalues[0], 50)
        x2_vals = np.linspace(self.minvalues[1], self.maxvalues[1], 50)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = np.array([[self.f(x1, x2) for x1, x2 in zip(x1_row, x2_row)] for x1_row, x2_row in zip(X1, X2)])

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
