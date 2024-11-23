#include <algorithm>
#include <chrono>
#include <format>
#include <iostream>
#include <random>

// BLAS C++ API
#include <cblas.h>
////

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

// RANDOM NUMBER GENERATION
std::random_device rd;
std::default_random_engine re = std::default_random_engine(rd());
std::uniform_real_distribution<double> urd =
    std::uniform_real_distribution<double>(0, 1);
float rng() { return (int)(urd(re) * 100) % 10; }
////

// PRINT MATRIX (FOR DEBUGGING)
void print_matrix(float *a, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int ii = 0; ii < cols; ii++) {
      std::cout << std::format("{} ", a[(i * cols) + ii]);
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
////

// NAIVE MATMUL
void gemm(float *a, long a_cols, long a_rows, float *b, long b_cols,
          long b_rows, float *c) {
  if (a_cols != b_rows) {
    std::cout << "Matrix dimensions do not match." << std::endl;
    return;
  }

  // ZERO OUT THE RETURN ARRAY
  memset(c, 0, a_rows * b_cols * sizeof(float));

  long c_cols = b_cols, c_rows = a_rows, depth = a_cols;

  long a_loc, b_loc, c_loc;

  for (int i = 0; i < c_rows; i++) {
    for (int ii = 0; ii < c_cols; ii++) {

      c_loc = (i * c_cols) + ii;

      for (int iii = 0; iii < depth; iii++) {
        a_loc = (i * depth) + iii;
        b_loc = (iii * c_cols) + ii;

        c[c_loc] += a[a_loc] * b[b_loc];
      }
    }
  }
}
////

int main() {

  int dim = 8192;
  int c_rows = dim, c_cols = dim, depth = dim;

  float *h_a = new float[depth * c_rows];
  float *h_b = new float[c_cols * depth];
  float *h_c = new float[c_cols * c_rows];

  // load a and b matrices with random values
  std::generate(h_a, h_a + (depth * c_rows), rng);
  std::generate(h_b, h_b + (c_cols * depth), rng);

  // std::cout << "MATRIX A" << std::endl;
  // print_matrix(a_p, c_rows, depth);

  // std::cout << "MATRIX B" << std::endl;
  // print_matrix(b_p, depth, c_cols);

  // timing counter
  double duration = 0;

  auto start = std::chrono::high_resolution_clock::now(),
       end = std::chrono::high_resolution_clock::now();

  long flops = (long)c_rows * (long)c_cols * (long)depth *
               2; /* flops used for one full matmul (1x mul, 1x add)*/
  // std::cout << flops << std::endl;

#define SKIP_NAIVE_GEMM
#ifndef SKIP_NAIVE_GEMM

  start = std::chrono::high_resolution_clock::now();
  gemm(h_a, dim, dim, h_b, dim, dim, h_c);
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();

  std::cout << std::format("GFLOPS: {} (Unoptimized)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(h_c, dim, dim);
  std::fill(h_c, h_c + (c_rows * c_cols), 0);

#endif

  // OPENBLAS SGEMM
  start = std::chrono::high_resolution_clock::now();

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1,
              &h_a[0], dim, &h_b[0], dim, 1, &h_c[0], dim);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();

  std::cout << std::format("GFLOPS: {} (OpenBLAS)", (flops / 1e9) / duration)
            << std::endl;
  // std::cout << "MATRIX C" << std::endl;
  // print_matrix(h_c, dim, dim);
  std::fill(h_c, h_c + (c_rows * c_cols), 0);
  ////

  // CUBLAS SGEMM
  cublasHandle_t handle;

  // pointers to device memory
  float *d_a, *d_b, *d_c;
  float alpha = 1.0f, beta = 0.0f;

  // allocate device memory
  cudaMalloc(&d_a, dim * dim * sizeof(float));
  cudaMalloc(&d_b, dim * dim * sizeof(float));
  cudaMalloc(&d_c, dim * dim * sizeof(float));

  // create cublas handle
  cublasCreate(&handle);

  start = std::chrono::high_resolution_clock::now();

  // copy data from host to device
  cublasSetMatrix(dim, dim, sizeof(float), h_a, dim, d_a, dim);
  cublasSetMatrix(dim, dim, sizeof(float), h_b, dim, d_b, dim);

  // call sgemm
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_b, dim,
              d_a, dim, &beta, d_c, dim);
  // cublas is column major, with no option for row major
  // luckily swapping matrices a and b has the same effect as transposition
  // https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication

  // copy result back to host
  cublasGetMatrix(dim, dim, sizeof(float), d_c, dim, h_c, dim);

  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (cuBLAS (with transfer))",
                           (flops / 1e9) / duration)
            << std::endl;
  ////

  // DO BENCHMARKING EXCLUDING THE HOST DEVICE TRANSFER STAGES
  // copy data from host to device
  cublasSetMatrix(dim, dim, sizeof(float), h_a, dim, d_a, dim);
  cublasSetMatrix(dim, dim, sizeof(float), h_b, dim, d_b, dim);

  // call sgemm
  start = std::chrono::high_resolution_clock::now();
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, dim, dim, dim, &alpha, d_b, dim,
              d_a, dim, &beta, d_c, dim);
  end = std::chrono::high_resolution_clock::now();

  // copy result back to host
  cublasGetMatrix(dim, dim, sizeof(float), d_c, dim, h_c, dim);

  duration = std::chrono::duration<double>(end - start).count();
  std::cout << std::format("GFLOPS: {} (cuBLAS (without transfer))",
                           (flops / 1e9) / duration)
            << std::endl;
  ////

  // clean up
  cublasDestroy(handle);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}