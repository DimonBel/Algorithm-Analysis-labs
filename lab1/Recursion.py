import time
import matplotlib.pyplot as plt


def nth_fibonacci(n):
    """
    Compute the nth Fibonacci number using a simple recursive approach.
    This method is inefficient for large n due to exponential time complexity.
    """
    if n <= 1:
        return n
    return nth_fibonacci(n - 1) + nth_fibonacci(n - 2)


def measure_fibonacci_performance():
    """
    Measure performance of Fibonacci calculation for specific numbers
    and create a performance visualization.
    """

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

    for n in test_numbers:
        start_time = time.perf_counter()
        fib_value = nth_fibonacci(n)
        end_time = time.perf_counter()

        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds

        print(f"Fibonacci({n}) = {fib_value}")
        print(f"Execution time: {execution_time:.6f} ms\n")

        execution_times.append(execution_time)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(
        test_numbers,
        execution_times,
        marker="o",
        linestyle="-",
        color="r",
        label="Execution Time",
    )
    plt.title("Fibonacci Calculation Performance (Naïve Recursion)")
    plt.xlabel("Fibonacci Number Index")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    measure_fibonacci_performance()
