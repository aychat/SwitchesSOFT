#include <stdio.h>					// STD INPUT-OUTPUT
#include <time.h>					// TIME LIBRARY
#include <stdlib.h>					// STANDARD LIBRARY
#include <math.h>					// C MATHS LIBRARY
#include <gsl/gsl_math.h>				// MATH LIBRARY UNDER GSL
#include <gsl/gsl_errno.h>				// ERROR CODES FOR GSL
#include <gsl/gsl_vector.h>				// GSL VECTOR OPERATIONS LIBRARY
#include <gsl/gsl_matrix.h>				// GSL MATRIX OPERATIONS LIBRARY
#include <gsl/gsl_blas.h>				// GSL LINEAR ALGEBRA LIBRARY
#include <gsl/gsl_odeiv2.h>				// GSL ODE-SOLVER
#include <gsl/gsl_sf.h>					// GSL SPECIAL FUNCTIONS
#include <gsl/gsl_complex_math.h>			// COMPLEX NUMBERS LIBRARY
#include <gsl/gsl_eigen.h>				// GSL EIGENSYSTEMS LIBRARY
#include <gsl/gsl_linalg.h>				// GSL LINEAR ALGEBRA LIBRARY
#include <gsl/gsl_rng.h>				// GSL RANDOM NUMBER GENERATOR


int print_complex_mat(gsl_matrix_complex *A)
{
	int mat_size = A->size1;
	int i,j;
	double x, y;
	for(i=0; i<mat_size; i++)
	{
		for(j=0; j<mat_size; j++)
		{
			x = GSL_REAL(gsl_matrix_complex_get(A,i,j));
			y = GSL_IMAG(gsl_matrix_complex_get(A,i,j));
			printf("(%3.2lf, %3.2lf) ", x, y);
//			printf("%3.2lf ", x);
		}
	printf("\n");
	}
	printf("\n\n");
	return GSL_SUCCESS;
}

int print_complex_vec(gsl_vector_complex *A)
{
	int vec_size = A->size;
	int i;
	double x, y;
	for(i=0; i<vec_size; i++)
	{
		x = GSL_REAL(gsl_vector_complex_get(A,i));
		y = GSL_IMAG(gsl_vector_complex_get(A,i));
		printf("(%3.3lf, %3.3lf) ", x, y);
//		printf("%3.2lf ", x);
	}
	printf("\n");
	return GSL_SUCCESS;
}



int main()
{
    gsl_matrix_complex *A = gsl_matrix_complex_calloc(2, 2);
    gsl_matrix_complex *B = gsl_matrix_complex_calloc(2, 2);
    gsl_matrix_complex *Bdag_B = gsl_matrix_complex_calloc(2, 2);
    gsl_matrix_complex *A_Bdag_B = gsl_matrix_complex_calloc(2, 2);
    gsl_matrix_complex *result = gsl_matrix_complex_calloc(2, 2);
    gsl_matrix_complex *result1 = gsl_matrix_complex_calloc(2, 2);

    const gsl_complex set_one = gsl_complex_rect(1.0, 0.0);
	const gsl_complex set_zero = gsl_complex_rect(0.0, 0.0);

	const gsl_rng_type * T;
    gsl_rng * r;
    gsl_rng_env_setup();
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    int i, j;
    gsl_vector_complex *random = gsl_vector_complex_calloc(8);
    for (i=0; i<8; i++)
    {
        gsl_vector_complex_set(random, i, gsl_complex_rect(gsl_rng_uniform (r), gsl_rng_uniform (r)));
    }

    for (i=0; i<2; i++)
        {
        for (j=0; j<2; j++)
            {
                gsl_matrix_complex_set(A, i, j, gsl_vector_complex_get(random, 2*i + j));
                gsl_matrix_complex_set(B, i, j, gsl_vector_complex_get(random, 4 + 2*i + j));
            }
        }

        gsl_matrix_complex_set(A, 1, 1, gsl_complex_sub(set_one, gsl_matrix_complex_get(A, 0, 0)));

    print_complex_mat(A);
    print_complex_mat(B);

	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, B, A, set_zero, result1);		// B_A
	gsl_blas_zgemm(CblasNoTrans, CblasTrans, set_one, result1, B, set_zero, result);		// B_A_B*

	print_complex_mat(result);

	gsl_blas_zgemm(CblasTrans, CblasNoTrans, set_one, B, B, set_zero, Bdag_B);	        // B*_B
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, Bdag_B, A, set_zero, A_Bdag_B);	// B*_B_A
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, A, Bdag_B, set_one, A_Bdag_B);	// B*_B_A + A_B*_B

	gsl_matrix_complex_scale(A_Bdag_B, gsl_complex_rect(-0.5, 0.0));
	gsl_matrix_complex_add(result, A_Bdag_B);
	
	print_complex_mat(result);
	
    return GSL_SUCCESS;
}