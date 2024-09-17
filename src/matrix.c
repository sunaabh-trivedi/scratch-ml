#include "matrix.h"
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include "autograd.h"

static Matrix* create_matrix(size_t rows, size_t cols)
{
    Matrix* ptr = (Matrix *)malloc(sizeof(Matrix));

    if(ptr == NULL) return NULL;

    ptr->rows = rows;
    ptr->cols = cols;

    ptr->data = (double *)_aligned_malloc(rows * cols * sizeof(double), 32);
    if(ptr->data == NULL) 
    {   
        free(ptr);
        return NULL;
    }

    return ptr;
}

Matrix* initialise_matrix(size_t rows, size_t cols)
{
    Matrix* mat = create_matrix(rows, cols);
    fill_matrix(mat, 0);

    return mat;
}

static Matrix* get_submatrix(const Matrix* matA, int row_start, int col_start, int size) 
{
    Matrix* sub = create_matrix(size, size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            sub->data[i * size + j] = matA->data[(row_start + i) * matA->cols + (col_start + j)];
        }
    }

    return sub;
}

void free_matrix(Matrix *matrix)
{
    _aligned_free(matrix->data);
    free(matrix);
}

int dot_product(const Matrix *matA, const Matrix *matB, double *res)
{   
    if(matA == NULL || matB == NULL) return 1;
    if(matA->cols != 1 || matB->cols != 1) return 2;
    if(matA->rows != matB->rows) return 3;

    *res = 0;
    for(size_t i = 0; i < matA->rows; i++)
    {
        *res += matA->data[i] * matB->data[i];
    }

    return 0;
}

int matrix_add(const Matrix *matA, const Matrix *matB, Matrix *res)
{
    if(matA == NULL || matB == NULL || res == NULL) return 1;
    if(matA->rows != matB->rows || matA->cols != matB ->cols || matA->rows != res->rows || matA->cols != res->cols) return 2;

    for(size_t i = 0; i < matA->rows * matA->cols; i++)
    {
        res->data[i] = matA->data[i] + matB->data[i];
    }

    gNode_t *gNode =  create_node(res, ADD, matA, matB);
    res->gNode = gNode;

    return 0;
}

int matrix_subtract(const Matrix *matA, const Matrix *matB, Matrix *res)
{
    if(matA == NULL || matB == NULL || res == NULL) return 1;
    if(matA->rows != matB->rows || matA->cols != matB ->cols || matA->rows != res->rows || matA->cols != res->cols) return 2;

    for(size_t i = 0; i < matA->rows * matA->cols; i++)
    {
        res->data[i] = matA->data[i] - matB->data[i];
    }

    gNode_t *gNode =  create_node(res, SUB, matA, matB);
    res->gNode = gNode;

    return 0;
}

int matrix_scalar_multiply(const Matrix *matA, double scalar, Matrix *res)
{
    if(matA == NULL || res == NULL) return 1;

    for(size_t i = 0; i < matA->rows * matA->cols; i++)
    {
        res->data[i] = scalar * matA->data[i];
    }

    gNode_t *gNode =  create_node(res, MUL, matA, NULL);
    res->gNode = gNode;

    return 0;
}

int matrix_multiply(const Matrix *matA, const Matrix *matB, Matrix *res)
{
    if(matA == NULL || matB == NULL || res == NULL) return 1;
    if(matA->cols != matB->rows || res->rows != matA->rows || res->cols != matB->cols) return 2;

    for(size_t i = 0; i < matA->rows; i++)
    {
        for(size_t j = 0; j < matB->cols; j++)
        {   
            double sum = 0.0f;
            for(size_t k = 0; k < matA->cols; k++)
            {
                sum += matA->data[i * matA->cols + k] * matB->data[k * matB->cols + j];
            }
            res->data[i * res->cols + j] = sum;
        }
    }

    gNode_t *gNode =  create_node(res, MUL, matA, matB);
    res->gNode = gNode;

    return 0;
}

static double __attribute__((always_inline)) reduce_add(__m256d vec)   {
    __m128d low = _mm256_extractf128_pd(vec, 0);
    __m128d high = _mm256_extractf128_pd(vec, 1);
    __m128d sum = _mm_add_pd(low, high);
    double result = _mm_cvtsd_f64(sum) + _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum));
    return result;
}

int matrix_multiply_opt(const Matrix *matA, const Matrix *matB, Matrix *res)
{
    if (matA == NULL || matB == NULL || res == NULL) return 1;
    if (matA->cols != matB->rows || res->rows != matA->rows || res->cols != matB->cols) return 2;

    Matrix *matBT = initialise_matrix(matB->cols, matB->rows);
    matrix_transpose(matB, matBT);

    #pragma omp parallel for
    for (size_t i = 0; i < matA->rows; i++) {
        for (size_t j = 0; j < matBT->rows; j++) {
            __m256d sum = _mm256_setzero_pd();
            size_t k;
            for (k = 0; k <= matA->cols - 4; k += 4) {
                __m256d a = _mm256_loadu_pd(&matA->data[i * matA->cols + k]);
                __m256d b = _mm256_loadu_pd(&matBT->data[j * matBT->cols + k]);
                sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
            }

            double partial_sum = reduce_add(sum);
            for (; k < matA->cols; k++) {
                partial_sum += matA->data[i * matA->cols + k] * matBT->data[j * matBT->cols + k]; // Do it the hard way for the remainder
            }

            res->data[i * res->cols + j] = partial_sum;
        }
    }

    free_matrix(matBT);

    return 0;
}

int matrix_transpose(const Matrix *matrix, Matrix *res)
{   
    if(matrix == NULL | res == NULL) return 1;
    if(matrix->rows != res->cols || matrix->cols != res->rows) return 2;

    for(int i = 0; i < matrix->rows; i++)
    {
        for(int j = 0; j < matrix->cols; j++)
        {
            res->data[i*res->cols + j] = matrix->data[j*res->cols + i];
        }
    }

    return 0;
}

int matrix_broadcast(const Matrix *src, Matrix *dest)
{
    if (src == NULL || dest == NULL) return 1;
    if (dest->rows % src->rows != 0 || dest->cols % src->cols != 0) return 2;

    #pragma omp parallel for
    for (size_t i = 0; i < dest->rows; i++) {
        for (size_t j = 0; j < dest->cols; j++) {
            dest->data[i * dest->cols + j] = src->data[(i % src->rows) * src->cols + (j % src->cols)];
        }
    }

    return 0;
}

static void split_matrix(const Matrix *matA, Matrix **matA11, Matrix **matA12, Matrix **matA21, Matrix **matA22)
{
    int n = matA->rows/2;

    *matA11 = get_submatrix(matA, 0, 0, n);
    *matA12 = get_submatrix(matA, 0, n, n);
    *matA21 = get_submatrix(matA, n, 0, n);
    *matA22 = get_submatrix(matA, n, n, n);
}

static void combine_matrix(Matrix *C11, Matrix *C12, Matrix *C21, Matrix *C22, Matrix *res)
{   
    int n = C11->rows;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res->data[i * res->cols + j] = C11->data[i * n + j];
            res->data[i * res->cols + (j + n)] = C12->data[i * n + j];
            res->data[(i + n) * res->cols + j] = C21->data[i * n + j];
            res->data[(i + n) * res->cols + (j + n)] = C22->data[i * n + j];
        }
    }

}

void fill_matrix(Matrix *matrix, double val)
{
    for(size_t i = 0; i < matrix->rows * matrix->cols; i++)
    {
        matrix->data[i] = val;
    }
}

void print_matrix(const Matrix* matrix)
{   
    if(matrix == NULL) return;

    printf("Matrix (%zu by %zu): \n", matrix->rows, matrix->cols);

    for(int i = 0; i < matrix->rows; i++)
    {   
        for(int j = 0; j < matrix->cols; j++)
        {
            printf("%f ", matrix->data[i*matrix->cols + j]);
        }

        printf("\n");
    }
}
