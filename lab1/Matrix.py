import time
import matplotlib.pyplot as plt

MOD = 10**9 + 7


def multiply(A, B):
    """
    Функция умножения двух 2x2 матриц с модульной арифметикой
    """
    C = [[0, 0], [0, 0]]

    C[0][0] = (A[0][0] * B[0][0] + A[0][1] * B[1][0]) % MOD
    C[0][1] = (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % MOD
    C[1][0] = (A[1][0] * B[0][0] + A[1][1] * B[1][0]) % MOD
    C[1][1] = (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % MOD

    A[0][0], A[0][1] = C[0][0], C[0][1]
    A[1][0], A[1][1] = C[1][0], C[1][1]


def power(M, expo):
    """
    Быстрое возведение матрицы в степень
    """
    ans = [[1, 0], [0, 1]]

    while expo:
        if expo & 1:
            multiply(ans, M)
        multiply(M, M)
        expo >>= 1

    return ans


def nthFibonacci(n):
    """
    Вычисление n-го числа Фибоначчи с помощью матричной экспоненциации
    """
    if n == 0 or n == 1:
        return 1

    M = [[1, 1], [1, 0]]
    F = [[1, 0], [0, 0]]

    res = power(M, n - 1)
    multiply(res, F)

    return res[0][0] % MOD


def measure_fibonacci_performance():
    """
    Измерение производительности вычисления чисел Фибоначчи
    и построение графика
    """
    test_numbers = [
        1,
        20,
        100,
        200,
        501,
        631,
        794,
        1000,
        1259,
        1585,
        1995,
        2512,
        3162,
        3981,
        5012,
        6310,
        7943,
        10000,
        12589,
        15849,
        20000,
    ]
    execution_times = []
    fibonacci_values = []

    for n in test_numbers:
        start_time = time.perf_counter()
        fib_value = nthFibonacci(n)
        end_time = time.perf_counter()

        execution_time = (end_time - start_time) * 1000  # Перевод в миллисекунды

        print(f"Fibonacci({n}) = {fib_value}")
        print(f"Execution time: {execution_time:.4f} ms\n")

        execution_times.append(execution_time)
        fibonacci_values.append(n)

    plt.figure(figsize=(10, 6))
    plt.plot(fibonacci_values, execution_times, marker="o")
    plt.title("Fibonacci Calculation Performance (Matrix Exponentiation)")
    plt.xlabel("Fibonacci Number Index")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    measure_fibonacci_performance()
