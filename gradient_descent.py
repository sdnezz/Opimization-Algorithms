import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class GradientDescent:
    def __init__(self):
        self.x0 = [0.5, 1.0]  # Начальная точка
        self.step_size = 0.1  # Шаг
        self.epsilon1 = 0.001  # Условие для градиента
        self.epsilon2 = 0.001  # Условие для точки и функции
        self.max_iter = 3     # Максимальное количество итераций

    def get_params(self):
        """Возвращает текущие параметры алгоритма"""
        return {
            "x0": self.x0.copy(),
            "step_size": self.step_size,
            "epsilon1": self.epsilon1,
            "epsilon2": self.epsilon2,
            "max_iter": self.max_iter
        }

    def set_params(self, params):
        """Устанавливает параметры алгоритма из словаря"""
        if "x0" in params:
            self.x0 = params["x0"].copy()
        if "step_size" in params:
            self.step_size = params["step_size"]
        if "epsilon1" in params:
            self.epsilon1 = params["epsilon1"]
        if "epsilon2" in params:
            self.epsilon2 = params["epsilon2"]
        if "max_iter" in params:
            self.max_iter = params["max_iter"]

    def plot(self, window):
        """Отрисовка графика градиентного спуска"""
        x_history, f_history, k = self.gradient(window)
        if not x_history:
            window.log_output("Ошибка: история точек пуста!")
            return

        x1_history = [point[0] for point in x_history]
        x2_history = [point[1] for point in x_history]

        window.log_output(
            f"Найденная точка минимума: x = {x1_history[-1]:.4f}, {x2_history[-1]:.4f}, f(x) = {f_history[-1]:.4f}, Итераций: {k + 1}")

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(x1_history, x2_history, 'b-', label='Траектория градиентного спуска')

        points = []
        for i, (x1, x2) in enumerate(zip(x1_history, x2_history)):
            radius = math.sqrt(x1 ** 2 + x2 ** 2)
            circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)
            if i == 0:
                point, = ax.plot(x1, x2, 'ro', markersize=10, label='Начальная точка (x^0)')
                points.append((point, x1, x2))
            elif i == len(x1_history) - 1:
                point, = ax.plot(x1, x2, 'k*', markersize=12, label='Точка минимума (x*)')
                points.append((point, x1, x2))
            else:
                point, = ax.plot(x1, x2, 'go', markersize=10, label='Промежуточная точка' if i == 1 else "")
                points.append((point, x1, x2))

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Градиентный спуск')
        ax.grid(True)
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=8)
        ax.set_aspect('equal', adjustable='box')

        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.5", fc="Pink", alpha=0.8),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(event):
            if event.inaxes != ax:
                annot.set_visible(False)
                canvas.draw_idle()
                return
            threshold = 0.05
            found = False
            for point, x, y in points:
                dist = math.sqrt((event.xdata - x) ** 2 + (event.ydata - y) ** 2)
                if dist < threshold:
                    annot.xy = (x, y)
                    annot.set_text(f"({x:.4f}, {y:.4f})")
                    annot.set_visible(True)
                    found = True
                    break
            if not found:
                annot.set_visible(False)
            canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", update_annot)
        canvas = FigureCanvas(fig)

        for i in reversed(range(window.graph_layout.count())):
            widget = window.graph_layout.itemAt(i).widget()
            if isinstance(widget, FigureCanvas):
                window.graph_layout.removeWidget(widget)
                widget.deleteLater()
        window.graph_layout.addWidget(canvas)
        canvas.draw()

    def f(self, x):
        return 2 * x[0] ** 2 + x[0] * x[1] + x[1] ** 2

    def grad_f(self, x):
        return [4 * x[0] + x[1], x[0] + 2 * x[1]]

    def next_point(self, x, grad, step_size):
        return [x[0] - step_size * grad[0], x[1] - step_size * grad[1]]

    def gradient(self, window):
        x_k = self.x0[:]
        k = 0
        x_history = [x_k.copy()]
        f_history = [self.f(x_k)]

        while k < self.max_iter:
            grad = self.grad_f(x_k)
            grad_norm = sum([g ** 2 for g in grad]) ** 0.5

            if grad_norm < self.epsilon1:
                window.log_output(f"Итерация {k}: x = {x_k}, f(x) = {self.f(x_k)} (Условие градиента выполнено)")
                break

            x_k1 = [x - self.step_size * g for x, g in zip(x_k, grad)]
            while self.f(x_k1) >= self.f(x_k):
                self.step_size /= 2
                x_k1 = [x - self.step_size * g for x, g in zip(x_k, grad)]

            if sum([(x1 - x0) ** 2 for x1, x0 in zip(x_k1, x_k)]) ** 0.5 < self.epsilon2 and abs(
                    self.f(x_k1) - self.f(x_k)) < self.epsilon2:
                window.log_output(
                    f"Итерация {k}: x = {x_k1}, f(x) = {self.f(x_k1)} (Условие разности целевой функции выполнено)")
                x_history.append(x_k1.copy())
                f_history.append(self.f(x_k1))
                break

            window.log_output(f"Итерация {k}: x = {x_k1}, f(x) = {self.f(x_k1)}")
            x_k = x_k1
            x_history.append(x_k.copy())
            f_history.append(self.f(x_k))
            k += 1

        if k == self.max_iter:
            window.log_output(
                f"Итерация {k}: x = {x_k}, f(x) = {self.f(x_k)} (достигнуто максимальное количество итераций)")

        return x_history, f_history, k