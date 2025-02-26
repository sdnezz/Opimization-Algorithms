import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class SimplexQuad:
    def __init__(self):
        # Параметры задачи оптимизации с дефолтными значениями
        self.c = [2, 1]  # Коэффициенты целевой функции
        self.A = [[1, 2], [1, 1]]  # Коэффициенты ограничений
        self.b = [2, 2]  # Правая часть ограничений
        self.max_iter = 100  # Максимальное количество итераций
        self.epsilon = 1e-6  # Точность для сходимости

    def plot(self,window):
        window.log_output(f"hello {self.max_iter}")