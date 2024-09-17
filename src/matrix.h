#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include "autograd.h"

typedef struct Matrix   {
    size_t rows;
    size_t cols;
    double *data;
    gNode_t *gNode;
}  Matrix;

Matrix* initialise_matrix(size_t rows, size_t cols);
void free_matrix(Matrix *matrix);

/* Arithmetic Functions */
int dot_product(const Matrix *A, const Matrix *B, double *res);
int matrix_add(const Matrix *A, const Matrix *B, Matrix *result);  
int matrix_subtract(const Matrix *A, const Matrix *B, Matrix *result);
int matrix_scalar_multiply(const Matrix *A, double scalar, Matrix *result);
int matrix_multiply(const Matrix *A, const Matrix *B, Matrix *result);
int matrix_multiply_opt(const Matrix *A, const Matrix *B, Matrix *res);
int matrix_transpose(const Matrix *A, Matrix *result);
int random_initialize(Matrix *matrix, double lower_bound, double upper_bound);

/* Utility Functions */
void fill_matrix(Matrix* matrix, double val);
void print_matrix(const Matrix* matrix);


#endif