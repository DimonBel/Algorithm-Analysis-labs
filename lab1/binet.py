import math
import time
import matplotlib.pyplot as plt


def fibonacci(n):
    """
    Compute the nth Fibonacci number using Binet's formula.

    Args:
        n (int): The index of the Fibonacci number to compute (0-based index)

    Returns:
        int: The nth Fibonacci number
    """
    # Validate input
    if n < 0:
        raise ValueError("Input must be a non-negative integer")

    # Special cases for 0 and 1
    if n <= 1:
        return n

    # Golden ratio (φ) and its negative counterpart (ψ)
    phi = (1 + math.sqrt(5)) / 2
    psi = (1 - math.sqrt(5)) / 2

    # Compute Fibonacci number using Binet's formula
    # Use round() to handle floating-point imprecision
    return round((math.pow(phi, n) - math.pow(psi, n)) / math.sqrt(5))


def measure_fibonacci_performance():
    """
    Measure performance of Fibonacci calculation for specific numbers
    and create a performance visualization.
    """
    # Список чисел для тестирования
    test_numbers = [
        1,
        20,
        40,
        70,
        100,
        130,
        140,
        200,
        220,
        250,
        270,
        330,
        370,
        400,
        440,
        460,
        500,
    ]

    # Списки для хранения результатов
    execution_times = []
    fibonacci_values = []

    # Измерение времени выполнения для каждого числа
    for n in test_numbers:
        start_time = time.perf_counter()
        fib_value = fibonacci(n)
        end_time = time.perf_counter()

        # Вычисление времени выполнения
        execution_time = (end_time - start_time) * 1000  # Перевод в миллисекунды

        print(f"Fibonacci({n}) = {fib_value}")
        print(f"Execution time: {execution_time:.4f} ms\n")

        # Сохранение результатов
        execution_times.append(execution_time)
        fibonacci_values.append(n)

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(fibonacci_values, execution_times, marker="o")
    plt.title("Fibonacci Calculation Performance")
    plt.xlabel("Fibonacci Number Index")
    plt.ylabel("Execution Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    measure_fibonacci_performance()
