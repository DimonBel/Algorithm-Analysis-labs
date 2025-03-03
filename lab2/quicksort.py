import matplotlib.pyplot as plt
import numpy as np
from random import randint
from timeit import repeat


def partition(array, low, high):
    pivot = array[high]
    i = low - 1
    for j in range(low, high):
        if array[j] <= pivot:
            i += 1
            array[i], array[j] = array[j], array[i]
    array[i + 1], array[high] = array[high], array[i + 1]
    return i + 1


def quickSort(array, low=0, high=None):
    if high is None:
        high = len(array) - 1
    if low < high:
        pi = partition(array, low, high)
        quickSort(array, low, pi - 1)
        quickSort(array, pi + 1, high)


def measure_time(algorithm, array):
    setup_code = f"from __main__ import {algorithm}"
    stmt = f"{algorithm}({array})"
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)
    return min(times)


sizes = [5000, 10000, 20000, 50000]
times = []

for size in sizes:
    array = [randint(0, 1000) for _ in range(size)]
    exec_time = measure_time("quickSort", array.copy())
    times.append(exec_time)

plt.figure(figsize=(10, 5))
plt.plot(sizes, times, marker="o", linestyle="-", color="b", label="Quicksort Time")
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Quicksort Execution Time vs Array Size")
plt.legend()
plt.grid()
plt.show()
