import matplotlib.pyplot as plt
import numpy as np
from random import randint
from timeit import repeat


def merge(arr1, arr2):
    result = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result


def merge_sort_iterative(arr):
    width = 1
    n = len(arr)
    while width < n:
        left = 0
        while left < n:
            mid = min(left + width, n)
            right = min(left + 2 * width, n)
            arr[left:right] = merge(arr[left:mid], arr[mid:right])
            left += 2 * width
        width *= 2
    return arr


def measure_time(algorithm, array):
    setup_code = f"from __main__ import {algorithm}"
    stmt = f"{algorithm}({array})"
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)
    return min(times)


sizes = [5000, 10000, 20000, 50000]

times = []

for size in sizes:
    array = [randint(0, 1000) for _ in range(size)]
    exec_time = measure_time("merge_sort_iterative", array.copy())
    times.append(exec_time)

plt.figure(figsize=(10, 5))
plt.plot(sizes, times, marker="o", linestyle="-", color="b", label="MergeSort Time")
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("MergeSort Execution Time vs Array Size")
plt.legend()
plt.grid()
plt.show()
