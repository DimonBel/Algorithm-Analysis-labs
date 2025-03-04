import matplotlib.pyplot as plt
import numpy as np
from random import randint
from timeit import repeat


def quicksort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        return quicksort(left) + [pivot] + quicksort(right)


def measure_time(algorithm, array):
    setup_code = f"from __main__ import {algorithm}"
    stmt = f"{algorithm}({array})"
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
    color="b",
    label="Quicksort Time with Comprehension",
)
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Quicksort Execution Time vs Array Size")
plt.legend()
plt.grid()
plt.show()
