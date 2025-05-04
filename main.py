from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLineEdit, QLabel, QTextEdit, QPushButton, QFormLayout
from PySide6.QtCore import Qt
from gradient_descent import GradientDescent
from simplex_quad import SimplexQuad
from genetic_algorithm import GeneticAlgorithm
from particle_swarm import ParticleSwarmOptimization
from bees_algorithm import BeesAlgorithm
import numpy as np
class GraphicalApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimization Algorithms")
        self.setGeometry(100, 100, 1300, 780)

        layout = QHBoxLayout()
        self.setLayout(layout)

        self.left_layout = QVBoxLayout()
        layout.addLayout(self.left_layout, 1)

        self.graph_layout = QVBoxLayout()
        layout.addLayout(self.graph_layout, 2)

        self.tabs = QTabWidget()
        self.left_layout.addWidget(self.tabs)

        self.start_button = QPushButton("Запустить алгоритм")
        self.start_button.setStyleSheet("background-color: #9b59b6; color: #fff; border-radius: 16px; padding: 8px; font-size: 14px; border: none; margin-top: 0px;")
        self.left_layout.addWidget(self.start_button)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; border: none;")
        self.left_layout.addWidget(self.output_text)

        self.load_styles()

        self.algorithms = [
            ("Градиентный спуск", GradientDescent()),
            ("Симплекс-квадратура", SimplexQuad()),
            ("Генетический алгоритм", GeneticAlgorithm()),  # Добавлен новый алгоритм
            ("Роевый алгоритм", ParticleSwarmOptimization()),
            ("Пчелиный алгоритм", BeesAlgorithm())
        ]
        self.saved_params = {i: {k: str(v) for k, v in algo.get_params().items()} for i, (_, algo) in enumerate(self.algorithms)}
        self.input_fields = {}
        self.create_tabs()
        self.create_tab_content(self.tabs.currentWidget())
        self.tabs.currentChanged.connect(self.on_tab_change)
        self.start_button.clicked.connect(self.run_algorithm)

    def load_styles(self):
        with open("styles.css", "r") as f:
            self.setStyleSheet(f.read())

    def create_tabs(self):
        for name, _ in self.algorithms:
            self.tabs.addTab(QWidget(), name)

    def on_tab_change(self, index):
        if self.tabs.currentIndex() < len(self.algorithms):
            self.saved_params[self.tabs.currentIndex()] = self.get_params_from_fields()
        self.create_tab_content(self.tabs.currentWidget())

    def get_params_from_fields(self):
        params = {}
        for param, field in self.input_fields.items():
            params[param] = field.text()
        return params

    def create_tab_content(self, tab):
        if tab.layout() is not None:
            while tab.layout().count():
                child = tab.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            QWidget().setLayout(tab.layout())

        tab_layout = QVBoxLayout()
        tab.setLayout(tab_layout)

        current_tab_index = self.tabs.currentIndex()
        if current_tab_index >= len(self.algorithms):
            label = QLabel("Алгоритм ещё не реализован")
            label.setAlignment(Qt.AlignCenter)
            tab_layout.addWidget(label)
            return

        _, self.algorithm = self.algorithms[current_tab_index]
        form_layout = QFormLayout()
        tab_layout.addLayout(form_layout)

        self.input_fields = {}
        params = self.saved_params.get(current_tab_index, {k: str(v) for k, v in self.algorithm.get_params().items()})
        for param, value in self.algorithm.get_params().items():
            label = QLabel(f"{param}:")
            # Преобразуем значение в строку явно
            input_field = QLineEdit(str(params.get(param, str(value))))
            form_layout.addRow(label, input_field)
            self.input_fields[param] = input_field

    def apply_fields_to_algorithm(self):
        if not self.input_fields:
            return
        params = {}
        for param, field in self.input_fields.items():
            value = field.text()
            try:
                # Обрабатываем параметры для разных алгоритмов
                if param in ["c", "A", "b", "func_structure", "ineq_signs"]:
                    # Преобразуем строки, представляющие списки, в настоящие списки
                    params[param] = eval(value) if value else self.algorithm.get_params()[param]
                elif param == "extr":
                    params[param] = value
                elif param == "max_iter":
                    params[param] = int(value)
                elif param == "genetic_bounds":  # Для генетического алгоритма bounds изменяем на genetic_bounds
                    params[param] = eval(value) if value else self.algorithm.get_params()[param]
                elif param == "bounds":  # Для SimplexQuad bounds остаётся так, как было
                    params[param] = eval(value) if value else self.algorithm.get_params()[param]
                elif param == "convergence_threshold":  # Обработка параметра convergence_threshold как float
                    params[param] = float(value) if value else self.algorithm.get_params()[param]
                elif param in ["initial_point", "minvalues", "maxvalues"]:
                    # Используем eval с пространством имён numpy
                    params[param] = eval(value, {"np": np}) if value else self.algorithm.get_params()[param]
                elif param in ["max_iterations", "swarmsize"]:
                    params[param] = int(value)
                elif param in ["current_velocity_ratio", "local_velocity_ratio", "global_velocity_ratio"]:
                    params[param] = float(value)
                elif param in ["scoutbeecount", "selectedbeecount", "bestbeecount", "selsitescount", "bestsitescount",
                               "max_iterations", "max_stagnation"]:
                    params[param] = int(value)
                elif param in ["range_lower", "range_upper", "range_shrink", "convergence_threshold"]:
                    params[param] = float(value)
                else:
                    # Преобразуем числа в float (или int, если целые числа)
                    params[param] = float(value) if '.' in value else int(value)
            except (ValueError, SyntaxError, NameError) as e:
                self.log_output(f"Ошибка в параметре {param}: {str(e)}")

        self.algorithm.set_params(params)

    def run_algorithm(self):
        self.log_output("Алгоритм запущен...")
        try:
            self.apply_fields_to_algorithm()
            self.log_output(f"Параметры перед запуском: {self.algorithm.__dict__}")
            self.algorithm.plot(self)
        except Exception as e:
            self.log_output(f"Ошибка при запуске: {str(e)}")

    def log_output(self, message):
        self.output_text.append(str(message))

if __name__ == "__main__":
    app = QApplication([])
    window = GraphicalApp()
    window.show()
    app.exec()