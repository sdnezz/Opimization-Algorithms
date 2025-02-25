from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLineEdit, QLabel, QTextEdit, \
    QPushButton, QFormLayout
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from gradient_descent import GradientDescent  # Подключаем алгоритм градиентного спуска

class GraphicalApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Optimization Algorithms")
        self.setGeometry(100, 100, 1300, 780)  # Размер окна

        # Главный макет
        layout = QHBoxLayout()
        self.setLayout(layout)

        # Слева - параметры алгоритмов и консоль
        self.left_layout = QVBoxLayout()
        layout.addLayout(self.left_layout, 1)

        # Справа - пространство для графика
        self.graph_layout = QVBoxLayout()
        layout.addLayout(self.graph_layout, 2)

        # Вкладки для алгоритмов
        self.tabs = QTabWidget()

        self.left_layout.addWidget(self.tabs)

        # Создание кнопки "Запустить алгоритм" (один раз)
        self.start_button = QPushButton("Запустить алгоритм")
        self.start_button.setStyleSheet("background-color: #9b59b6; color: #fff; border-radius: 16px; padding: 8px; font-size: 14px; border: none; margin-top: 0px;")
        self.left_layout.addWidget(self.start_button)

        # Вывод результата (консоль)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setStyleSheet("background-color: #2b2b2b; color: #ffffff; border: none;")
        self.left_layout.addWidget(self.output_text)
        self.load_styles()
        # Словарь для хранения ссылок на поля ввода
        self.input_fields = {}
        #создаем вкладки, в этой функции вручную добавляем новые
        self.create_tabs()
        #затем на основе имеющихся вкладок смотрим, какой алгоритм на текущей вкладке
        #и присваиваем полю алгоритма текущий алгоритм выбранной вкладки
        self.algorithm = self.select_alg()
        # Создаем поля для ввода параметров для выбранного алгоритма
        self.create_tab_content(self.tabs.currentWidget())
        # Связываем переключение на другую вкладку с обновлением поля класса алгоритм
        #КОННЕКТЫ-ОБРАБОТЧИКИ ВНУТРИ ВЬЮХИ
        self.tabs.currentChanged.connect(self.on_tab_change)
        self.start_button.clicked.connect(self.run_algorithm)

    def load_styles(self):
        with open("styles.css", "r") as f:
            self.setStyleSheet(f.read())

    def create_tabs(self):
        """Создание просто вкладок с названиями"""
        self.tabs.addTab(QTabWidget(), "Градиентный спуск")
        self.tabs.addTab(QTabWidget(), "Новая вкладка")
        self.tabs.addTab(QTabWidget(), "Новая вкладка")
        self.tabs.addTab(QTabWidget(), "Новая вкладка")
        self.tabs.addTab(QTabWidget(), "Новая вкладка")
        self.tabs.addTab(QTabWidget(), "Новая вкладка")

    def select_alg(self):
        """Выбор алгоритма в зависимости от активной вкладки"""
        current_tab_index = self.tabs.currentIndex()  # Получаем индекс текущей вкладки
        # Если выбрана первая вкладка, используем GradientDescent
        if current_tab_index == 0:
            return GradientDescent()

    #меняем в поле класса текущий алгоритм
    def on_tab_change(self, index):
        self.algorithm = self.select_alg()
        self.create_tab_content(self.tabs.currentWidget())

    def create_tab_content(self, tab):
        """Создание контента вкладки для текущего алгоритма"""
        tab_layout = QVBoxLayout()  # Создаем макет для вкладки
        tab.setLayout(tab_layout)
        form_layout = QFormLayout()  # Макет для полей ввода
        tab_layout.addLayout(form_layout)

        # Проходим по всем атрибутам класса алгоритма
        for param, value in self.algorithm.__dict__.items():
            # Для каждого параметра создаем QLabel и QLineEdit
            label = QLabel(f"{param}:")  # Метка для каждого параметра
            input_field = QLineEdit(str(value))  # Поле для ввода с значением по умолчанию из алгоритма
            form_layout.addRow(label, input_field)  # Добавляем метку и поле ввода в макет
            # Добавляем поле ввода в словарь
            self.input_fields[param] = input_field

            # Привязка изменения значения поля к обновлению параметров
            input_field.textChanged.connect(self.params_update)

    #при нажатии кнопки "стартуем" запускается алгоритм с обновленными параметрами
    def params_update(self):
        """Обновляем параметры алгоритма в зависимости от введенных значений в поля ввода."""
        for param, field in self.input_fields.items():
            # Получаем значение из поля ввода
            value = field.text()

            # Проверка, является ли параметр списком (например, для x0)
            if isinstance(getattr(self.algorithm, param), list):
                setattr(self.algorithm, param, [float(x) for x in value.strip('[]').split(',')])
            else:
                setattr(self.algorithm, param, float(value))  # Для простых числовых значений

    def plot_update(self):
        """Обновление графика после запуска алгоритма"""
        self.algorithm.plot(self)  # Вызов метода plot из алгоритма

    def run_algorithm(self):
        """Запуск выбранного алгоритма"""
        self.log_output("Алгоритм запущен...")
        try:
            # Обновление параметров перед запуском
            self.params_update()
            # Запуск алгоритма и получение результата
            result = self.plot_update()
            # Выводим результат в консоль
            self.log_output(result)

        except ValueError:
            self.log_output("Ошибка: введите верные значения переменных!")

    #Вывод в консоль
    def log_output(self, message):
        self.output_text.append(message)

if __name__ == "__main__":
    app = QApplication([])
    window = GraphicalApp()
    window.show()
    app.exec()
