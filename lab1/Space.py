import time
import matplotlib.pyplot as plt


def nth_fibonacci(n):
    if n <= 1:
        return n

    curr = 0
    prev1 = 1
    prev2 = 0

    for _ in range(2, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr

    return curr


def measure_fibonacci_performance():
    test_numbers = [
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
    ]
    execution_times = []
    fibonacci_values = []

    for n in test_numbers:
        start_time = time.perf_counter()
        fib_value = nth_fibonacci(n)
        end_time = time.perf_counter()

        execution_time = (end_time - start_time) * 1000

        print(f"Fibonacci({n}) = {fib_value}")
        print(f"Execution time: {execution_time:.4f} ms\n")

        execution_times.append(execution_time)
        fibonacci_values.append(n)

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
