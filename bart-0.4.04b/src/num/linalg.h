/* Copyright 2013-2015. The Regents of the University of California.
 * Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 */ 

#include <complex.h>
#include <stdbool.h>

#ifdef __cplusplus
#error This file does not support C++
#endif

extern void mat_identity(int A, int B, complex float x[A][B]);
extern void mat_zero(int A, int B, complex float x[A][B]);
extern void mat_gaussian(int A, int B, complex float x[A][B]);
extern void mat_mul(int A, int B, int C, complex float x[A][C], const complex float y[A][B], const complex float z[B][C]);

#define MVLA(x) __restrict__ (A)
extern void mat_muladd(int A, int B, int C, complex float x[MVLA(A)][C], const complex float y[MVLA(A)][B], const complex float z[MVLA(B)][C]);
extern void mat_add(int A, int B, complex float x[A][B], const complex float y[A][B], const complex float z[A][B]);
extern void mat_transpose(int A, int B, complex float dst[B][A], const complex float src[A][B]);
extern void mat_adjoint(int A, int B, complex float dst[B][A], const complex float src[A][B]);
extern void mat_conj(int A, int B, complex float dst[B][A], const complex float src[A][B]);
extern void mat_copy(int A, int B, complex float dst[A][B], const complex float src[A][B]);
extern bool mat_inverse(unsigned int N, complex float dst[N][N], const complex float src[N][N]);
extern void mat_vecmul(unsigned int A, unsigned int B, complex float out[A], const complex float mat[A][B], const complex float in[B]);
extern void mat_kron(unsigned int A, unsigned int B, unsigned int C, unsigned int D,
		complex float out[A * C][B * D], const complex float in1[A][B], const complex float in2[C][D]);
extern void mat_vec(unsigned int A, unsigned int B, complex float out[A * B], const complex float in[A][B]);
extern void vec_mat(unsigned int A, unsigned int B, complex float out[A][B], const complex float in[A * B]);

// extern complex double vec_dot(int N, const complex float x[N], const complex float y[N]);
extern complex float vec_dot(int N, const complex float x[N], const complex float y[N]);
extern void vec_saxpy(int N, complex float x[N], complex float alpha, const complex float y[N]);
extern void gram_matrix(int N, complex float cov[N][N], int L, const complex float data[N][L]);
extern void gram_schmidt(int M, int N, float val[N], complex float vecs[M][N]);
extern void gram_matrix2(int N, complex float cov[N * (N + 1) / 2], int L, const complex float data[N][L]);
extern void pack_tri_matrix(int N, complex float cov[N * (N + 1) / 2], const complex float m[N][N]);
extern void unpack_tri_matrix(int N, complex float m[N][N], const complex float cov[N * (N + 1) / 2]);
extern void orthiter_noinit(int M, int N, int iter, float vals[M], complex float out[M][N], const complex float matrix[N][N]);
extern void orthiter(int M, int N, int iter, float vals[M], complex float out[M][N], const complex float matrix[N][N]);
extern void cholesky(int N, complex float A[N][N]);
extern void cholesky_solve(int N, complex float x[N], const complex float L[N][N], const complex float b[N]);
extern void cholesky_double(int N, complex double A[N][N]);
extern void cholesky_solve_double(int N, complex double x[N], const complex double L[N][N], const complex double b[N]);
extern complex float vec_mean(long D, const complex float src[D]);
extern void vec_axpy(long N, complex float x[N], complex float alpha, const complex float y[N]);
extern void vec_sadd(long D, complex float alpha, complex float dst[D], const complex float src[D]);
extern void thomas_algorithm(int N, complex float f[N], const complex float A[N][3], const complex float d[N]);


