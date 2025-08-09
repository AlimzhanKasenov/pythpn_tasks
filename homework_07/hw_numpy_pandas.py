# I. NUMPY
import matplotlib
matplotlib.use('TkAgg')  # фикс для PyCharm ошибки FigureCanvasInterAgg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Одномерный массив размера 10, заполненный нулями, пятый элемент равен 1
arr = np.zeros(10, dtype=int)
arr[4] = 1
arr_2d = arr.reshape(2, 5)
print("Массив 2D (с пятым элементом = 1):\n", arr_2d)

# 2. Одномерный массив со значениями от 10 до 49, развернуть, найти четные элементы
arr2 = np.arange(10, 50)
arr2_rev = arr2[::-1]
even_elements = arr2_rev[arr2_rev % 2 == 0]
print("\nРазвернутый массив:\n", arr2_rev)
print("Четные элементы:\n", even_elements)

# 3. Двумерный массив 3x3 со значениями от 0 до 8
arr3 = np.arange(9).reshape(3, 3)
print("\nМассив 3x3:\n", arr3)

# 4. Массив 4x3x2 со случайными значениями, найти минимум и максимум
arr4 = np.random.random((4, 3, 2))
print("\nМассив 4x3x2:\n", arr4)
print("Минимум:", arr4.min(), "Максимум:", arr4.max())

# 5. Матричное умножение массивов 6x4 и 4x3
A = np.random.randint(1, 10, (6, 4))
B = np.random.randint(1, 10, (4, 3))
mat_mult = np.dot(A, B)
print("\nМатрица A:\n", A)
print("Матрица B:\n", B)
print("Результат умножения:\n", mat_mult)

# 6. Случайный массив 7x7, среднее и СКО, нормализация
arr5 = np.random.rand(7, 7)
mean = arr5.mean()
std = arr5.std()
arr5_norm = (arr5 - mean) / std
print("\nСреднее:", mean, "Ст. отклонение:", std)
print("Нормализованный массив:\n", arr5_norm)

# -----------------------------------------------------
# II. PANDAS

# Загружаем датасет tips
tips = sns.load_dataset("tips")

# Первые 5 строк
print("\nПервые 5 строк:\n", tips.head())

# Размер (строки, колонки)
print("\nРазмер данных:", tips.shape)

# Пропуски
print("\nПропуски в данных:\n", tips.isnull().sum())

# Описание числовых признаков
print("\nОписание числовых признаков:\n", tips.describe())

# Максимальное total_bill
print("\nМакс total_bill:", tips["total_bill"].max())

# Количество курящих
print("\nКоличество курящих:\n", tips["smoker"].value_counts())

# Средний total_bill в зависимости от day (добавил observed=True для фикса FutureWarning)
print("\nСредний total_bill по дням:\n", tips.groupby("day", observed=True)["total_bill"].mean())

# Отбор по total_bill > медианы и средний tip по sex (тоже observed=True)
median_tb = tips["total_bill"].median()
filtered = tips[tips["total_bill"] > median_tb]
print("\nСредний tip по sex (total_bill > медианы):\n", filtered.groupby("sex", observed=True)["tip"].mean())

# smoker -> бинарный
tips["smoker_bin"] = tips["smoker"].map({"No": 0, "Yes": 1})
print("\nsmoker_bin:\n", tips[["smoker", "smoker_bin"]].head())

# -----------------------------------------------------
# III. ВИЗУАЛИЗАЦИЯ

# 1. Гистограмма total_bill
plt.figure(figsize=(6, 4))
sns.histplot(tips["total_bill"], kde=True, bins=20)
plt.title("Распределение total_bill")
plt.show()

# 2. Scatterplot total_bill vs tip
plt.figure(figsize=(6, 4))
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("total_bill vs tip")
plt.show()

# 3. Pairplot
sns.pairplot(tips)
plt.show()

# 4. total_bill vs day
plt.figure(figsize=(6, 4))
sns.boxplot(data=tips, x="day", y="total_bill")
plt.title("total_bill по дням")
plt.show()

# 5. Две гистограммы tip по time
plt.figure(figsize=(6, 4))
sns.histplot(data=tips, x="tip", hue="time", kde=True)
plt.title("Распределение tip по time")
plt.show()

# 6. Два scatterplot для Male и Female, цвет по smoker
g = sns.FacetGrid(tips, col="sex", hue="smoker", height=4)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip")
g.add_legend()
plt.show()

# -----------------------------------------------------
# IV. ВЫВОДЫ
print("""
Выводы:
1. Средний чек (total_bill) выше в выходные дни (Sat, Sun), чем в будние.
2. Чаевые (tip) растут с увеличением суммы счета, но не линейно.
3. Курящие клиенты часто оставляют немного меньше чаевых.
4. Женщины и мужчины оставляют чаевые в среднем на одинаковом уровне, но мужчины чаще имеют более высокие счета.
""")
