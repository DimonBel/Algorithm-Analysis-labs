import matplotlib.pyplot as plt
import numpy as np
from random import randint
from timeit import repeat


def partition2(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


def quicksort(arr, low, high):
    if low < high:
        pi = partition2(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)


def measure_time(algorithm, array):
    setup_code = f"from __main__ import {algorithm}"
    stmt = f"{algorithm}({array}, 0, len({array}) - 1)"
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)
    return min(times)


sizes = [5000, 10000, 20000, 50000]
times = []

for size in sizes:
    array = [randint(0, 1000) for _ in range(size)]
    exec_time = measure_time("quicksort", array.copy())
    times.append(exec_time)

plt.figure(figsize=(10, 5))
plt.plot(
    sizes,
    times,
    marker="o",
    linestyle="-",
    color="r",
    label="QuickSort Time with inplaceQuick",
)
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("QuickSort Execution Time vs Array Size")
plt.legend()
plt.grid()
plt.show()
