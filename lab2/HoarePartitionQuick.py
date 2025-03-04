import matplotlib.pyplot as plt
import numpy as np
from random import randint
from timeit import repeat


def partition(arr, low, high):
    pivot = arr[low]
    i = low - 1
    j = high + 1
    while True:
        i += 1
        while arr[i] < pivot:
            i += 1
        j -= 1
        while arr[j] > pivot:
            j -= 1
        if i >= j:
            return j
        arr[i], arr[j] = arr[j], arr[i]


def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi)
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
    label="QuickSort Time with HoarePartitionQuick",
)
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("QuickSort Execution Time vs Array Size")
plt.legend()
plt.grid()
plt.show()
