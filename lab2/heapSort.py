# def heapify(arr, n, i):
#     largest = i  # Initialize largest as root
#     l = 2 * i + 1  # left = 2*i + 1
#     r = 2 * i + 2  # right = 2*i + 2

#     # See if left child of root exists and is
#     # greater than root

#     if l < n and arr[i] < arr[l]:
#         largest = l

#     # See if right child of root exists and is
#     # greater than root

#     if r < n and arr[largest] < arr[r]:
#         largest = r

#     # Change root, if needed

#     if largest != i:
#         (arr[i], arr[largest]) = (arr[largest], arr[i])  # swap

#         # Heapify the root.

#         heapify(arr, n, largest)


# # The main function to sort an array of given size


# def heapSort(arr):
#     n = len(arr)

#     # Build a maxheap.
#     # Since last parent will be at (n//2) we can start at that location.

#     for i in range(n // 2, -1, -1):
#         heapify(arr, n, i)

#     # One by one extract elements

#     for i in range(n - 1, 0, -1):
#         (arr[i], arr[0]) = (arr[0], arr[i])  # swap
#         heapify(arr, i, 0)


# # Driver code to test above

# # arr = [
# #     12,
# #     11,
# #     13,
# #     5,
# #     6,
# #     7,
# # ]
# # heapSort(arr)
# # n = len(arr)
# # print("Sorted array is")
# # for i in range(n):
# #     print(arr[i])

# # This code is contributed by Mohit Kumra
# from random import randint
# from timeit import repeat


# def run_sorting_algorithm(algorithm, array):
#     # Set up the context and prepare the call to the specified
#     # algorithm using the supplied array. Only import the
#     # algorithm function if it's not the built-in `sorted()`.
#     setup_code = f"from __main__ import {algorithm}" if algorithm != "sorted" else ""

#     stmt = f"{algorithm}({array})"

#     # Execute the code ten different times and return the time
#     # in seconds that each execution took
#     times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)

#     # Finally, display the name of the algorithm and the
#     # minimum time it took to run
#     print(f"Algorithm: {algorithm}. Minimum execution time: {min(times)}")


# ARRAY_LENGTH = 10000

# if __name__ == "__main__":
#     # Generate an array of `ARRAY_LENGTH` items consisting
#     # of random integer values between 0 and 999
#     array = [randint(0, 1000) for i in range(ARRAY_LENGTH)]

#     # Call the function using the name of the sorting algorithm
#     # and the array you just created
#     run_sorting_algorithm(algorithm="heapSort", array=array)


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
