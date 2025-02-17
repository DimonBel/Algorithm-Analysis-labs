import time
import matplotlib.pyplot as plt

# Initialize Fibonacci array
FibArray = [0, 1]


def fibonacci(n):
    if n < 0:
        raise ValueError("Incorrect input")

    if n < len(FibArray):
        return FibArray[n]
    else:
        temp_fib = fibonacci(n - 1) + fibonacci(n - 2)
        FibArray.append(temp_fib)
        return temp_fib


def measure_fibonacci_performance():
    """
    Measure performance of Fibonacci calculation for specific numbers
    and create a performance visualization.
    """
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

    execution_times = []

    for n in test_numbers:
        start_time = time.perf_counter()
        fib_value = fibonacci(n)
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
        color="b",
        label="Execution Time",
    )
    plt.title("Fibonacci Calculation Performance (Memoization with List)")
    plt.xlabel("Fibonacci Number Index")
    plt.ylabel("Execution Time (ms)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    measure_fibonacci_performance()
