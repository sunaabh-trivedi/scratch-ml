#include "matrix.h"
#include <time.h>
#include <stdio.h>

int main(void)
{
    Matrix *matA = initialise_matrix(1024, 1024);
    Matrix *matB = initialise_matrix(1024, 1024);
    Matrix *res = initialise_matrix(1024, 1024);

    fill_matrix(matA, 8923.2131238123);
    fill_matrix(matB, 91283912.123123);
    
    for(int i = 0; i < 10; i++)
    {
        clock_t begin = clock();
        int ret = matrix_multiply(matA, matB, res);
        clock_t end = clock();
        double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

        if(ret) {   
            printf("Error code %i\n", ret);
            return 1;
        }

        // print_matrix(res);
        printf("Time spent in matmul: %fs\n", time_spent);
    }

    free_matrix(matA);
    free_matrix(matB);
    free_matrix(res);

    return 0;
}