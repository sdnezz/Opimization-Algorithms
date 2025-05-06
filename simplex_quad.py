# simplex_quad.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import plotly.graph_objects as go
from PySide6.QtWidgets import QVBoxLayout, QWidget
import plotly.io as pio
from PySide6.QtWebEngineWidgets import QWebEngineView

class SimplexQuad:
    def __init__(self, c=None, A=None, b=None, bounds=None, extr="min", func_structure=None, ineq_signs=None):
        self.extr = extr
        self.invert = -1 if extr == "max" else 1

        self.c = c if c is not None else [2, 2, 2, -4, -6]
        self.default_structure = ["x1^2", "x1*x2", "x2^2", "x1", "x2"]
        self.func_structure = func_structure if func_structure is not None else self.default_structure

        self.A = A if A is not None else [[1, 2]]
        self.b = b if b is not None else [2]
        self.ineq_signs = ineq_signs if ineq_signs is not None else ["<="] * len(self.b)
        self.bounds = bounds if bounds is not None else [(0, None), (0, None)]

    def get_params(self):
        """Возвращает параметры для отображения в интерфейсе"""
        return {
            "c": self.c,
            "A": self.A,
            "b": self.b,
            "bounds": self.bounds,
            "extr": self.extr,
            "func_structure": self.func_structure,
            "ineq_signs": self.ineq_signs
        }

    def set_params(self, params):
        """Устанавливает параметры из интерфейса"""
        if "c" in params:
            self.c = params["c"]
        if "A" in params:
            self.A = params["A"]
        if "b" in params:
            self.b = params["b"]
        if "bounds" in params:
            self.bounds = params["bounds"]
        if "extr" in params:
            self.extr = params["extr"]
            self.invert = -1 if self.extr == "max" else 1
        if "func_structure" in params:
            self.func_structure = params["func_structure"]
        if "ineq_signs" in params:
            self.ineq_signs = params["ineq_signs"]

    def objective(self, x):
        """Функция цели"""
        x1, x2 = x
        terms = {
            "x1^2": x1 ** 2,
            "x1*x2": x1 * x2,
            "x2^2": x2 ** 2,
            "x1": x1,
            "x2": x2
        }
        return self.invert * sum(self.c[i] * terms[self.func_structure[i]] for i in range(len(self.c)))

    def constraint(self, x):
        """Обрабатываем все знаки ограничений"""
        constraints = []
        for i in range(len(self.b)):
            if self.ineq_signs[i] == "<=":
                constraints.append({'type': 'ineq', 'fun': lambda x, xbi=i: self.b[i] - sum(a_ij * x_j for a_ij, x_j in zip(self.A[i], x))})
            elif self.ineq_signs[i] == ">=":
                constraints.append({'type': 'ineq', 'fun': lambda x, xbi=i: sum(a_ij * x_j for a_ij, x_j in zip(self.A[i], x)) - self.b[i]})
            elif self.ineq_signs[i] == "=":
                constraints.append({'type': 'eq', 'fun': lambda x, xbi=i: sum(a_ij * x_j for a_ij, x_j in zip(self.A[i], x)) - self.b[i]})
        return constraints

    def plot(self, window):
        """Запускает оптимизацию и рисует 3D-график в интерфейсе"""
        x0 = [0.5, 0.5]  # Начальная точка
        method = 'trust-constr' if self.extr == "max" else 'SLSQP'
        result = minimize(self.objective, x0, method=method, constraints=self.constraint(x0), bounds=self.bounds)

        if result.success:
            solution = result.x
            f_optimal = self.invert * result.fun
            output_message = f"Оптимальное решение: x1 = {solution[0]:.4f}, x2 = {solution[1]:.4f}\n" \
                             f"{'Минимальное' if self.extr == 'min' else 'Максимальное'} значение F(x): {f_optimal:.4f}"
            window.log_output(output_message)
            self._plot_3d_interface(window, solution)
        else:
            window.log_output("Решение не найдено!")

    def _plot_3d_interface(self, window, solution):
        """3D-график через Plotly"""
        x1_vals = np.linspace(0, 2, 50)
        x2_vals = np.linspace(0, 2, 50)
        X1, X2 = np.meshgrid(x1_vals, x2_vals)
        Z = np.array([[self.objective([x1, x2]) for x1, x2 in zip(x1_row, x2_row)] for x1_row, x2_row in zip(X1, X2)])

        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X1, y=X2, colorscale='plasma'))
        fig.add_trace(go.Scatter3d(
            x=[solution[0]], y=[solution[1]], z=[self.objective(solution)],
            mode='markers',
            marker=dict(size=6, color='red'),
            name="Оптимум"
        ))

        fig.update_layout(
            title="3D график целевой функции",
            scene=dict(
                xaxis_title="x1",
                yaxis_title="x2",
                zaxis_title="F(x)",
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