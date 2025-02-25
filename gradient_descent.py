import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# Класс для алгоритма градиентного спуска
class GradientDescent:
    def __init__(self):
        # Параметры алгоритма с начальными значениями
        self.x0 = [0.5, 1.0]  # Начальная точка
        self.step_size = 0.1  # Шаг
        self.epsilon1 = 0.001 # Условие для градиента
        self.epsilon2 = 0.001  # Условие для точки и функции
        self.max_iter = 3  # Максимальное количество итераций

    def plot(self, window):
        """Отрисовка графика градиентного спуска с траекторией, окружностями и всплывающими координатами"""
        # Запуск алгоритма градиентного спуска и получение истории
        x_history, f_history, k = self.gradient(window)

        # Проверка на пустую историю
        if not x_history:
            window.log_output("Ошибка: история точек пуста!")
            return

        # Преобразуем x_history в два списка для x1 и x2
        x1_history = [point[0] for point in x_history]
        x2_history = [point[1] for point in x_history]

        # Вывод результата
        window.log_output(
            f"Найденная точка минимума: x = {x1_history[-1]:.4f}, {x2_history[-1]:.4f}, f(x) = {f_history[-1]:.4f}, Итераций: {k + 1}")

        # Создаем фигуру и оси для графика с увеличенным размером
        fig, ax = plt.subplots(figsize=(10, 8))

        # Рисуем траекторию градиентного спуска (синяя линия)
        ax.plot(x1_history, x2_history, 'b-', label='Траектория градиентного спуска')

        # Список для хранения объектов точек
        points = []

        # Рисуем точки и окружности
        for i, (x1, x2) in enumerate(zip(x1_history, x2_history)):
            radius = math.sqrt(x1 ** 2 + x2 ** 2)
            circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--', alpha=0.5)
            ax.add_patch(circle)

            # Рисуем точку и сохраняем её в список
            if i == 0:
                point, = ax.plot(x1, x2, 'ro', markersize=10, label='Начальная точка (x^0)')
                points.append((point, x1, x2))
            elif i == len(x1_history) - 1:
                point, = ax.plot(x1, x2, 'k*', markersize=12, label='Точка минимума (x*)')
                points.append((point, x1, x2))
            else:
                point, = ax.plot(x1, x2, 'go', markersize=10, label='Промежуточная точка' if i == 1 else "")
                points.append((point, x1, x2))

        # Настройки графика
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Градиентный спуск')
        ax.grid(True)
        ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0), fontsize=8)
        ax.set_aspect('equal', adjustable='box')

        # Создаем аннотацию, которая будет отображаться при наведении
        annot = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                            bbox=dict(boxstyle="round,pad=0.5", fc="Pink", alpha=0.8),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)  # Изначально скрыта

        # Функция для обновления аннотации при наведении
        def update_annot(event):
            if event.inaxes != ax:
                annot.set_visible(False)
                canvas.draw_idle()
                return

            # Радиус чувствительности (в единицах данных)
            threshold = 0.05  # Можно настроить
            found = False

            for point, x, y in points:
                # Проверяем расстояние от курсора до точки
                dist = math.sqrt((event.xdata - x) ** 2 + (event.ydata - y) ** 2)
                if dist < threshold:  # Если курсор близко к точке
                    annot.xy = (x, y)
                    annot.set_text(f"({x:.4f}, {y:.4f})")
                    annot.set_visible(True)
                    found = True
                    break

            if not found:  # Если курсор не над точкой
                annot.set_visible(False)

            canvas.draw_idle()

        # Подключаем обработчик событий мыши
        fig.canvas.mpl_connect("motion_notify_event", update_annot)

        # Создаем объект FigureCanvas для интеграции с Qt
        canvas = FigureCanvas(fig)

        # Очищаем старое содержимое и добавляем новый график
        for i in reversed(range(window.graph_layout.count())):
            widget = window.graph_layout.itemAt(i).widget()
            if isinstance(widget, FigureCanvas):
                window.graph_layout.removeWidget(widget)
                widget.deleteLater()
        window.graph_layout.addWidget(canvas)
        canvas.draw()

    def plot_ellipse(self, ax, center, f_val):
        """Построение эллипса для заданного уровня функции f(x) = C"""
        # Для функции f(x) = 2x1^2 + x1x2 + x2^2, эллипсы можно аппроксимировать
        # Используем параметрическое уравнение эллипса: x1 = a*cos(t), x2 = b*sin(t)
        # Где a и b определяются из формы эллипса, зависящей от f_val
        theta = [i * (2 * math.pi / 100) for i in range(101)]  # 101 точка для полного круга
        x_ellipse = []
        y_ellipse = []

        # Коэффициенты для эллипса, основанные на функции f(x) = 2x1^2 + x1x2 + x2^2
        # Приближенно решаем для эллипса, предполагая, что минимум в (0,0), и эллипс масштабируется
        a = math.sqrt(f_val / 2)  # Примерное масштабирование по x1 (учитываем 2x1^2)
        b = math.sqrt(f_val)  # Примерное масштабирование по x2 (учитываем x2^2)

        for t in theta:
            x1 = center[0] + a * math.cos(t)  # Центрируем эллипс относительно центра
            x2 = center[1] + b * math.sin(t)
            # Корректируем для перекоса (x1x2), добавляя небольшое смещение
            x_ellipse.append(x1 - 0.1 * x1 * x2 / f_val)  # Простая аппроксимация перекоса
            y_ellipse.append(x2)

        ax.plot(x_ellipse, y_ellipse, 'k-', alpha=0.5)  # Рисуем эллипс (черным)
        # Добавляем метки уровней, как в изображении
        if f_val == 0.5:
            ax.text(max(x_ellipse) * 1.1, max(y_ellipse) * 0.9, 'f(x) = 0.5', fontsize=8)
        elif f_val == 1.0:
            ax.text(max(x_ellipse) * 1.1, max(y_ellipse) * 0.9, 'f(x) = 1.0', fontsize=8)
        elif f_val == 2.0:
            ax.text(max(x_ellipse) * 1.1, max(y_ellipse) * 0.9, 'f(x) = 2.0', fontsize=8)
        elif f_val == 3.0:
            ax.text(max(x_ellipse) * 1.1, max(y_ellipse) * 0.9, 'f(x) = 3.0', fontsize=8)

    # Функция f(x)
    def f(self, x):
        return 2 * x[0] ** 2 + x[0] * x[1] + x[1] ** 2

    # Градиент функции f(x)
    def grad_f(self, x):
        return [4 * x[0] + x[1], x[0] + 2 * x[1]] #частные пр-ые перем., указ-ет направление возрастания

    # Метод для вычисления следующей точки
    def next_point(self, x, grad, step_size):
        return [x[0] - step_size * grad[0], x[1] - step_size * grad[1]]

    # Метод градиентного спуска с постоянным шагом
    def gradient(self, window):
        x_k = self.x0[:]  # Копия начальной точки
        k = 0
        x_history = [x_k.copy()]  # Сохраняем начальную точку
        f_history = [self.f(x_k)]

        while k < self.max_iter:
            grad = self.grad_f(x_k)
            #корень (из (чпроизв.x1)^2+(чпроизв.x2)^2)
            grad_norm = sum([g ** 2 for g in grad]) ** 0.5  # Норма градиента

            #  если норма градиента меньше eps1
            if grad_norm < self.epsilon1:
                window.log_output(f"Итерация {k}: x = {x_k}, f(x) = {self.f(x_k)} (Условие градиента выполнено)")
                break  # Выходим из цикла, все точки уже сохранены

            #xk+1=xk−step_size⋅∇f(xk)
            x_k1 = [x - self.step_size * g for x, g in zip(x_k, grad)]
            while self.f(x_k1) >= self.f(x_k):
                self.step_size /= 2
                x_k1 = [x - self.step_size * g for x, g in zip(x_k, grad)]

            if sum([(x1 - x0) ** 2 for x1, x0 in zip(x_k1, x_k)]) ** 0.5 < self.epsilon2 and abs(
                    self.f(x_k1) - self.f(x_k)) < self.epsilon2:
                window.log_output(
                    f"Итерация {k}: x = {x_k1}, f(x) = {self.f(x_k1)} (Условие разности целевой функции выполнено)")
                x_history.append(x_k1.copy())  # Добавляем конечную точку
                f_history.append(self.f(x_k1))
                break  # Выходим из цикла

            window.log_output(f"Итерация {k}: x = {x_k1}, f(x) = {self.f(x_k1)}")
            x_k = x_k1
            x_history.append(x_k.copy())  # Сохраняем каждую промежуточную точку
            f_history.append(self.f(x_k))
            k += 1

        # Если достигнуто максимальное количество итераций
        if k == self.max_iter:
            window.log_output(
                f"Итерация {k}: x = {x_k}, f(x) = {self.f(x_k)} (достигнуто максимальное количество итераций)")

        return x_history, f_history, k