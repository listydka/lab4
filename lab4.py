import numpy as np
import matplotlib.pyplot as plt

# Ввод данных
K = int(input("Введите K="))
N = int(input("Введите N="))

# Проверка, что N четное
if N % 2 != 0:
    print("Ошибка: N должно быть четным числом.")
    exit()

# Инициализация подматриц с целенаправленным заполнением
D = np.array(np.random.randint(-10, 10, (N//2, N//2)))
E = np.array(np.random.randint(-10, 10, (N//2, N//2)))
C = np.array(np.random.randint(-10, 10, (N//2, N//2)))
B = np.array(np.random.randint(-10, 10, (N//2, N//2)))

# Формируем матрицу A
A = np.hstack((np.vstack((D, C)), np.vstack((E, B))))
print("Матрица A:\n", A, "\n")

# Разделяем C на нечетные и четные столбцы для подсчета нулевых элементов
Ccut1 = C[:, 0::2]  # Четные столбцы (индексы 0, 2, 4, ...)
Ccut2 = C[:, 1::2]  # Нечетные столбцы (индексы 1, 3, 5, ...)

# Находим количество нулевых элементов в этих столбцах
Ccut1_zeros = np.sum(Ccut1 == 0)
Ccut2_zeros = np.sum(Ccut2 == 0)

# Печатаем количество нулевых элементов
print(f"Количество нулевых в четных столбцах C: {Ccut1_zeros}")
print(f"Количество нулевых в нечетных столбцах C: {Ccut2_zeros}")

# Условие: если нулевых в нечетных столбцах больше, то меняем C и B симметрично,
# иначе C и E меняем местами несимметрично
if Ccut1_zeros > Ccut2_zeros:
    C = np.flip(C, axis=1)
    B = np.flip(B, axis=1)
    F = np.hstack((np.vstack((D, E)), np.vstack((C, B))))
else:
    F = np.hstack((np.vstack((D, C)), np.vstack((E, B))))

print(f"\nМатрица F:\n {F}\n")

# Проверяем условие для вычисления результата
det_A = np.linalg.det(A)
sum_diag_F = np.sum(np.diagonal(F)) + np.sum(np.fliplr(F).diagonal())  # Сумма диагональных элементов матрицы F

print(f"Определитель матрицы A: {det_A}")
print(f"Сумма диагональных элементов матрицы F: {sum_diag_F}")

if det_A > sum_diag_F:
    # Вычисляем выражение A^(-1) * A^T - K * F
    A_inv = np.linalg.inv(A)
    A_T = np.transpose(A)
    G = np.dot(A_inv, A_T) - K * F
    print("Результат выражения A^(-1) * A^T - K * F:\n", G, "\n")
else:
    # Вычисляем выражение (A^T + G^(-1) - F^(-1)) * K
    A_T = np.transpose(A)
    G_inv = np.linalg.inv(np.tril(A))  # Нижняя треугольная матрица из A и её обратная
    F_inv = np.linalg.inv(F)
    result = np.dot((A_T + G_inv - F_inv), K)
    print("Результат выражения (A^T + G^(-1) - F^(-1)) * K:\n", result, "\n")

# Графики
plt.title("Зависимости: y = sin от элементов F, x = cos от элементов F")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.plot(np.cos(F), np.sin(F), linestyle="--", color="r")
plt.show()

plt.title("Высота столбца от числа элемента первой строки")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.bar(range(0, ((N // 2) * 2)), F[0], color='r', alpha=0.9)
plt.show()

plt.title("Соответствие номера и квадрата элемента из первой строки")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.scatter(range(0, ((N // 2) * 2)), F[0])
plt.show()
