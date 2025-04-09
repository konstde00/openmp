#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream> 
#include <vector>
#include <omp.h>
#include <chrono>

namespace py = pybind11;

typedef std::vector<std::vector<double>> Matrix;

Matrix add_matrices(const Matrix& A, const Matrix& B) {
    size_t rows = A.size(), cols = A[0].size();
    Matrix C(rows, std::vector<double>(cols, 0.0));
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(rows); i++) {
        for (int j = 0; j < static_cast<int>(cols); j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

Matrix subtract_matrices(const Matrix& A, const Matrix& B) {
    size_t rows = A.size(), cols = A[0].size();
    Matrix C(rows, std::vector<double>(cols, 0.0));
    
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < static_cast<int>(rows); i++) {
        for (int j = 0; j < static_cast<int>(cols); j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

Matrix multiply_by_scalar(const Matrix& A, double scalar) {
    size_t rows = A.size(), cols = A[0].size();
    Matrix C(rows, std::vector<double>(cols, 0.0));
    
    for (int i = 0; i < static_cast<int>(rows); i++) {
        #pragma omp parallel for firstprivate(i)
        for (int j = 0; j < static_cast<int>(cols); j++) {
            C[i][j] = A[i][j] * scalar;
        }
    }
    return C;
}

Matrix transpose_matrix(const Matrix& A) {
    size_t rows = A.size(), cols = A[0].size();
    Matrix C(cols, std::vector<double>(rows, 0.0));
    
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < static_cast<int>(rows); i++) {
        for (int j = 0; j < static_cast<int>(cols); j++) {
            C[j][i] = A[i][j];
        }
    }
    return C;
}

PYBIND11_MODULE(omp_matrix, m) {
    m.doc() = "Parallel matrix operations using OpenMP";
    m.def("add_matrices", [](const Matrix& A, const Matrix& B) {
        auto start = std::chrono::high_resolution_clock::now();
        Matrix C = add_matrices(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        return py::make_tuple(C, duration);
    });
    m.def("subtract_matrices", [](const Matrix& A, const Matrix& B) {
        auto start = std::chrono::high_resolution_clock::now();
        Matrix C = subtract_matrices(A, B);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        return py::make_tuple(C, duration);
    });
    m.def("multiply_by_scalar", [](const Matrix& A, double scalar) {
        auto start = std::chrono::high_resolution_clock::now();
        Matrix C = multiply_by_scalar(A, scalar);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        return py::make_tuple(C, duration);
    });
    m.def("transpose_matrix", [](const Matrix& A) {
        auto start = std::chrono::high_resolution_clock::now();
        Matrix C = transpose_matrix(A);
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end - start).count();
        return py::make_tuple(C, duration);
    });
}
