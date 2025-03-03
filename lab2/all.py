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


def merge(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left = arr[:mid]
        right = arr[mid:]
        merge(left)
        merge(right)
        i = j = k = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1


def pigeonhole_sort(a):
    my_min = min(a)
    my_max = max(a)
    size = my_max - my_min + 1
    holes = [0] * size
    for x in a:
        holes[x - my_min] += 1
    i = 0
    for count in range(size):
        while holes[count] > 0:
            holes[count] -= 1
            a[i] = count + my_min
            i += 1


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


def heap_sort(arr):
    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(arr, i, 0)


def measure_time(algorithm, array):
    stmt = lambda: globals()[algorithm](array.copy())
    times = repeat(stmt, repeat=3, number=10)
    return min(times)


sizes = [100, 200, 500, 1000, 5000, 10000, 20000, 50000]
sorting_algorithms = [
    "quickSort",
    "merge",
    "pigeonhole_sort",
    "heap_sort",
]
times = {algo: [] for algo in sorting_algorithms}

for size in sizes:
    array = [randint(0, 1000) for _ in range(size)]
    for algo in sorting_algorithms:
        exec_time = measure_time(algo, array.copy())
        times[algo].append(exec_time)

plt.figure(figsize=(10, 5))
for algo in sorting_algorithms:
    plt.plot(sizes, times[algo], marker="o", linestyle="-", label=f"{algo} Time")
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Sorting Algorithms Execution Time vs Array Size")
plt.legend()
plt.grid()
plt.show()

# Print execution times
for algo in sorting_algorithms:
    print(f"Execution times for {algo}: {times[algo]}")
