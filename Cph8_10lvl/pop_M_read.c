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

int print_mat(gsl_matrix *A)
{
	int mat_size = A->size1;
	int i,j;
	double x, y;

	FILE * fp;
	fp = fopen ("file1.txt", "w+");

	for(i=0; i<mat_size; i++)
	{
		for(j=0; j<mat_size; j++)
		{	
			x = gsl_matrix_get(A,i,j);
			fprintf(fp, "%3.3e ", x);
			printf("%3.3e ", x);
		}
	fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");
	return GSL_SUCCESS;
}

int main()
{
	gsl_matrix *pop_M = gsl_matrix_alloc(10, 10);
	FILE *pop_mat = fopen ("pop_M.out", "rb");
	gsl_matrix_fread(pop_mat, pop_M);
	fclose (pop_mat);

	print_mat(pop_M);

}
