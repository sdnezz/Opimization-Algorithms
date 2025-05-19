import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from PySide6.QtWebEngineWidgets import QWebEngineView
from bacterial_foraging import BacterialForagingOptimization
from particle_swarm import ParticleSwarmOptimization
import time

class HybridBFOPSO:
    def __init__(self):
        self.bfo_iterations = 2
        self.pso_iterations = 50
        self.num_bacteria = 50
        self.swarmsize = 50
        self.bounds_lower = -5
        self.bounds_upper = 5
        self.dimension = 2
        self.chem_steps = 100
        self.repro_steps = 4
        self.step_size = 0.1
        self.elim_prob = 0.25
        self.elim_count = 10
        self.current_velocity_ratio = 0.5
        self.local_velocity_ratio = 2.0
        self.global_velocity_ratio = 2.0
        self.f = lambda x, y: x**2 + y**2

    def get_params(self):
        return {
            "bfo_iterations": self.bfo_iterations,
            "pso_iterations": self.pso_iterations,
            "num_bacteria": self.num_bacteria,
            "swarmsize": self.swarmsize,
            "bounds_lower": self.bounds_lower,
            "bounds_upper": self.bounds_upper,
            "dimension": self.dimension,
            "chem_steps": self.chem_steps,
            "repro_steps": self.repro_steps,
            "step_size": self.step_size,
            "elim_prob": self.elim_prob,
            "elim_count": self.elim_count,
            "current_velocity_ratio": self.current_velocity_ratio,
            "local_velocity_ratio": self.local_velocity_ratio,
            "global_velocity_ratio": self.global_velocity_ratio
        }

    def set_params(self, params):
        if "bfo_iterations" in params:
            self.bfo_iterations = params["bfo_iterations"]
        if "pso_iterations" in params:
            self.pso_iterations = params["pso_iterations"]
        if "num_bacteria" in params:
            self.num_bacteria = params["num_bacteria"]
        if "swarmsize" in params:
            self.swarmsize = params["swarmsize"]
        if "bounds_lower" in params:
            self.bounds_lower = params["bounds_lower"]
        if "bounds_upper" in params:
            self.bounds_upper = params["bounds_upper"]
        if "dimension" in params:
            self.dimension = params["dimension"]
        if "chem_steps" in params:
            self.chem_steps = params["chem_steps"]
        if "repro_steps" in params:
            self.repro_steps = params["repro_steps"]
        if "step_size" in params:
            self.step_size = params["step_size"]
        if "elim_prob" in params:
            self.elim_prob = params["elim_prob"]
        if "elim_count" in params:
            self.elim_count = params["elim_count"]
        if "current_velocity_ratio" in params:
            self.current_velocity_ratio = params["current_velocity_ratio"]
        if "local_velocity_ratio" in params:
            self.local_velocity_ratio = params["local_velocity_ratio"]
        if "global_velocity_ratio" in params:
            self.global_velocity_ratio = params["global_velocity_ratio"]

    def run(self):
        times = []
        start_time = time.time()

        bfo = BacterialForagingOptimization()
        bfo_params = {
            "num_bacteria": self.num_bacteria,
            "chem_steps": self.chem_steps,
            "repro_steps": self.repro_steps,
            "elim_steps": self.bfo_iterations,
            "step_size": self.step_size,
            "elim_prob": self.elim_prob,
            "elim_count": self.elim_count,
            "bounds_lower": self.bounds_lower,
            "bounds_upper": self.bounds_upper,
            "dimension": self.dimension
        }
        bfo.set_params(bfo_params)
        bfo_best, bfo_trajectory, bfo_status, bfo_log, top_positions, bfo_times, bfo_value = bfo.run()

        times.extend([(i, t + (time.time() - start_time)) for i, t in bfo_times])

        pso = ParticleSwarmOptimization(
            max_iterations=self.pso_iterations,
            swarmsize=self.swarmsize,
            minvalues=[self.bounds_lower] * self.dimension,
            maxvalues=[self.bounds_upper] * self.dimension,
            current_velocity_ratio=self.current_velocity_ratio,
            local_velocity_ratio=self.local_velocity_ratio,
            global_velocity_ratio=self.global_velocity_ratio,
            initial_positions=top_positions
        )
        pso_best, pso_trajectory, pso_status, pso_log, pso_times, pso_value = pso.run()

        times.extend([(i + len(bfo_times), t + (time.time() - start_time)) for i, t in pso_times])

        trajectory = bfo_trajectory + pso_trajectory
        iterations_log = bfo_log + pso_log
        return pso_best, trajectory, "Гибридный BFO+PSO завершён", iterations_log, times, pso_value

    def plot(self, window):
        final_point, trajectory, message, iterations_log, times, final_value = self.run()
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
        fig.update_layout(title="Траектория гибридного BFO+PSO", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="f(x, y)", camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.5, y=1.5, z=1.5))), margin=dict(l=0, r=0, b=0, t=30))
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