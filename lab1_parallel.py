import random
import time
import os
import string
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

#CPU-bound

# Monte Carlo PI
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x, y = random.random(), random.random()
        if x*x + y*y <= 1:
            inside += 1
    return inside

def pi_sequential(iterations):
    start = time.time()
    inside = monte_carlo_pi(iterations)
    pi = 4 * inside / iterations
    end = time.time()
    print(f"~PI sequential~ PI={pi:.6f} Time={end-start:.2f}s")
    return end-start

def pi_worker(n):
    return monte_carlo_pi(n)

def pi_parallel(iterations, workers):
    start = time.time()
    chunk = iterations // workers
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(pi_worker, [chunk]*workers))
    pi = 4 * sum(results) / iterations
    end = time.time()
    print(f"~PI parallel~ Workers={workers} PI={pi:.6f} Time={end-start:.2f}s")
    return end-start

# Prime numbers
def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0: return False
    return True

def primes_worker(numbers):
    return [n for n in numbers if is_prime(n)]

def primes_sequential(start, end):
    start_time = time.time()
    primes = [n for n in range(start, end) if is_prime(n)]
    end_time = time.time()
    print(f"~Primes sequential~ Found={len(primes)} Time={end_time-start_time:.2f}s")
    return end_time-start_time

def primes_parallel(start, end, workers):
    numbers = list(range(start, end))
    chunks = np.array_split(numbers, workers)
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(primes_worker, chunks))
    primes = [p for sub in results for p in sub]
    end_time = time.time()
    print(f"~Primes parallel~ Workers={workers} Found={len(primes)} Time={end_time-start_time:.2f}s")
    return end_time-start_time

# Factorization
def factorize(n):
    factors = []
    d = 2
    while d*d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1: factors.append(n)
    return factors

def factorization_worker(numbers):
    return [factorize(n) for n in numbers]

def factorization_sequential(numbers):
    start = time.time()
    _ = [factorize(n) for n in numbers]
    end = time.time()
    print(f"~Factorization sequential~ Time={end-start:.2f}s")
    return end-start

def factorization_parallel(numbers, workers):
    chunks = np.array_split(numbers, workers)
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(factorization_worker, chunks))
    # flatten results
    flattened = [f for sub in results for f in sub]
    end = time.time()
    print(f"~Factorization parallel~ Workers={workers} Time={end-start:.2f}s")
    return end-start

# ---------------- Memory-bound ----------------
def transpose_sequential(matrix):
    start = time.time()
    result = matrix.T
    end = time.time()
    print(f"~Matrix transpose sequential~ Time={end-start:.6f}s")
    return end-start

def transpose_worker(chunk):
    return chunk.T

def transpose_parallel(matrix, workers):
    chunks = np.array_split(matrix, workers, axis=0)
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(transpose_worker, chunks))
    result = np.vstack(results)
    end = time.time()
    print(f"~Matrix transpose parallel~ Workers={workers} Time={end-start:.4f}s")
    return end-start

#IO-bound
def generate_files(folder, num_files, words_per_file):
    os.makedirs(folder, exist_ok=True)
    for i in range(num_files):
        with open(f"{folder}/file{i}.txt", "w") as f:
            words = ["".join(random.choices(string.ascii_lowercase, k=5)) for _ in range(words_per_file)]
            f.write(" ".join(words))
    return folder

def count_words_in_file(path):
    with open(path, "r") as f:
        return len(f.read().split())

def count_words_sequential(folder):
    start = time.time()
    total = sum(count_words_in_file(os.path.join(root, file))
                for root, _, files in os.walk(folder) for file in files)
    end = time.time()
    print(f"~Word count sequential~ Words={total} Time={end-start:.2f}s")
    return end-start

def count_words_parallel(folder, workers):
    files = [os.path.join(root, file)
            for root, _, files in os.walk(folder) for file in files]
    start = time.time()
    total = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(count_words_in_file, f) for f in files]
        for future in as_completed(futures):
            total += future.result()
    end = time.time()
    print(f"~Word count parallel~ Workers={workers} Words={total} Time={end-start:.2f}s")
    return end-start

#main
if __name__ == "__main__":
    workers = 4

    print("~CPU bound tasks~")
    pi_seq_time = pi_sequential(10_000_000)
    pi_par_time = pi_parallel(10_000_000, workers)

    primes_seq_time = primes_sequential(1, 50_000)
    primes_par_time = primes_parallel(1, 50_000, workers)

    numbers = [random.randint(10**6, 10**7) for _ in range(1_000)]
    factor_seq_time = factorization_sequential(numbers)
    factor_par_time = factorization_parallel(numbers, workers)

    print("~Memory bound task~")
    matrix = np.random.rand(2_000, 2_000)
    mem_seq_time = transpose_sequential(matrix)
    mem_par_time = transpose_parallel(matrix, workers)

    print("~IO bound task~")
    folder = generate_files("/tmp/lab1_data", num_files=500, words_per_file=200)
    io_seq_time = count_words_sequential(folder)
    io_par_time = count_words_parallel(folder, workers)

    #graphs
    tasks = [
        ('PI', pi_seq_time, pi_par_time),
        ('Primes', primes_seq_time, primes_par_time),
        ('Factorization', factor_seq_time, factor_par_time),
        ('Matrix', mem_seq_time, 0),
        ('Word count', io_seq_time, io_par_time)
    ]

    for label, seq, par in tasks:
        plt.figure(figsize=(6,4))
        plt.bar(['Sequential','Parallel'], [seq, par if par>0 else seq], color=['blue','orange'])
        plt.title(f"Speed comparison: {label}")
        plt.ylabel("Time")
        plt.show()