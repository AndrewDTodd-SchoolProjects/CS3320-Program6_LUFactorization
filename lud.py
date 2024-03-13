import numpy as np
import sys

def factor(A, n, pivot):
    for i in range(n):
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if np.isclose(A[max_row, i], 0):
            return -1
        A[[i, max_row]] = A[[max_row, i]]
        pivot[i] = max_row
        for j in range(i+1, n):
            A[j, i] /= A[i, i]
            for k in range(i+1, n):
                A[j, k] -= A[j, i] * A[i, k]
    
    return 0

def solve(A, n, pivot, b, x):
    b_copy = np.copy(b)

    for i in range(n):
        if i != pivot[i]:
            b_copy[i], b_copy[pivot[i]] = b_copy[pivot[i]], b_copy[i]

    y = np.zeros(n)
    for i in range(n):
        y[i] = b_copy[i] - np.dot(A[i, :i], y[:i])

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

def main():
    for filename in sys.argv[1:]:
        try:
            with open(filename, 'r') as file:
                n = int(file.readline())
                A = np.array([list(map(float, file.readline().split())) for _ in range(n)])
                pivot = np.zeros(n, dtype=int)
                status = factor(A, n, pivot)
                if status == -1:
                    print(f"A matrix in {filename} is singular")
                    continue

                np.set_printoptions(suppress=True)
                print(f"\nFile {filename}")
                print(f"L/U = \n{np.array2string(A, precision=2, separator=', ', floatmode='fixed')}")
                num_rhs = int(file.readline())
                for _ in range(num_rhs):
                    b = np.array(list((map(float, file.readline().split()))))
                    x = np.zeros_like(b)
                    solve(A, n, pivot, b, x)
                    print(f"b = {np.array2string(b, precision=2, separator=', ', floatmode='fixed')}")
                    print(f"x = {np.array2string(x, precision=2, separator=', ', floatmode='fixed')}")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")

if __name__ == "__main__":
    main()