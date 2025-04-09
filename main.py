#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import random
import omp_matrix

def generate_random_matrix(rows, cols, min_val=0, max_val=100):
    return [[random.randint(min_val, max_val) for _ in range(cols)] for _ in range(rows)]

def measure_operation_time(operation, *args, name):
    result, elapsed_time = operation(*args)
    elapsed_time /= 1000  # Convert milliseconds to seconds
    print(f"[{name}] completed in {elapsed_time:.4f} s")
    return result

def print_header():
    print("PYBIND11 OPENMP MATRIX OPERATIONS")
    print("Operations: Addition | Subtraction | Scalar Multiplication | Transposition")

def main():
    sizes = [(1000, 1000), (2000, 2000), (5000, 5000)]
    scalar = 5
    
    print_header()
    print()
    
    for rows, cols in sizes:
        print(f"Processing matrices of size {rows}x{cols}")
        
        matrix_a = generate_random_matrix(rows, cols)
        matrix_b = generate_random_matrix(rows, cols)
        
        measure_operation_time(omp_matrix.add_matrices, matrix_a, matrix_b, name="Matrix Addition")
        measure_operation_time(omp_matrix.subtract_matrices, matrix_a, matrix_b, name="Matrix Subtraction")
        measure_operation_time(omp_matrix.multiply_by_scalar, matrix_a, scalar, name="Scalar Multiplication")
        measure_operation_time(omp_matrix.transpose_matrix, matrix_a, name="Matrix Transposition")
        
    print()
    print("Finished matrix operations with OpenMP and Pybind11.")

if __name__ == "__main__":
    main()
