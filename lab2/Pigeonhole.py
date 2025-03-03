import matplotlib.pyplot as plt
import numpy as np
from random import randint
from timeit import repeat


def pigeonhole_sort(a):
    my_min = min(a)
    my_max = max(a)
    size = my_max - my_min + 1
    holes = [0] * size
    for x in a:
        assert type(x) is int, "integers only please"
        holes[x - my_min] += 1
    i = 0
    for count in range(size):
        while holes[count] > 0:
            holes[count] -= 1
            a[i] = count + my_min
            i += 1


def measure_time(algorithm, array):
    setup_code = f"from __main__ import {algorithm}"
    stmt = f"{algorithm}({array})"
    times = repeat(setup=setup_code, stmt=stmt, repeat=3, number=10)
    return min(times)


sizes = [5000, 10000, 20000, 50000]
times = []

for size in sizes:
    array = [randint(0, 1000) for _ in range(size)]
    exec_time = measure_time("pigeonhole_sort", array.copy())
    times.append(exec_time)

plt.figure(figsize=(10, 5))
plt.plot(
    sizes, times, marker="o", linestyle="-", color="b", label="Pigeonhole Sort Time"
)
plt.xlabel("Array Size")
plt.ylabel("Execution Time (seconds)")
plt.title("Pigeonhole Sort Execution Time vs Array Size")
plt.legend()
plt.grid()
plt.show()
