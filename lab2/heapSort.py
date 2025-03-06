import matplotlib.pyplot as plt
import numpy as np
from random import randint
from timeit import repeat


def heapify(arr, n, i):
    largest = i
    l = 2 * i + 1
    r = 2 * i + 2

    if l < n and arr[i] < arr[l]:
        largest = l
    if r < n and arr[largest] < arr[r]:
        largest = r

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)


def heapSort(arr):
    n = len(arr)
    for i in range(n // 2, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


def measure_time(algorithm, array):
    setup_code = f"from __main__ import {algorithm}"
    stmt = f"{algorithm}({array})"
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)
    return min(times)


sizes = [5000, 10000, 20000, 50000]

times = []

for size in sizes:
    array = [randint(0, 1000) for _ in range(size)]
    exec_time = measure_time("heapSort", array.copy())
    times.append(exec_time)

plt.figure(figsize=(10, 5))
plt.plot(sizes, times, marker="o", linestyle="-", color="b", label="HeapSort Time")
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("HeapSort Execution Time vs Array Size")
plt.legend()
plt.grid()
plt.show()
