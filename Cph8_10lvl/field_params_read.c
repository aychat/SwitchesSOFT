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


int print_vec(const gsl_vector *A)
{
	const int vec_size = A->size;
	int i;
	FILE * fp;

	fp = fopen ("file.txt", "w+");


	for(i=0; i<vec_size; i++)
	{
		fprintf(fp, "%.8e \n", gsl_vector_get(A,i) );
	}
	fprintf(fp, "\n\n");

	for(i=0; i<vec_size; i++)
	{
		printf("%.8e \n", gsl_vector_get(A,i) );
	}
	printf("\n\n");

	fclose(fp);
	
	return GSL_SUCCESS;
}

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
		}
	fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");
	return GSL_SUCCESS;
}

int main()
{
	int Nfreq = 4;
	gsl_vector *A_phi_params = gsl_vector_alloc(2*Nfreq);
	FILE *field_f = fopen ("field_out_params.out", "rb");
	gsl_vector_fread(field_f, A_phi_params);
	fclose (field_f);

	print_vec(A_phi_params);

	gsl_vector *field_ini = gsl_vector_alloc(250);
	FILE *f_ini = fopen ("field_ini.out", "rb");
	gsl_vector_fread(f_ini, field_ini);
	fclose (f_ini);

	gsl_vector *field_fin = gsl_vector_alloc(250);
	FILE *f_fin = fopen ("field_out.out", "rb");
	gsl_vector_fread(f_fin, field_fin);
	fclose (f_fin);

	gsl_matrix *pop_t = gsl_matrix_alloc(10, 100000);
	FILE *pop_dyn = fopen ("pop_dynamical.out", "rb");
	gsl_matrix_fread(pop_dyn, pop_t);
	fclose (pop_dyn);

	gsl_matrix *pop_M = gsl_matrix_alloc(10, 10);
	FILE *pop_mat = fopen ("pop_M.out", "rb");
	gsl_matrix_fread(pop_mat, pop_M);
	fclose (pop_mat);

	print_mat(pop_M);

	int i, j;
	
	FILE *f = fopen ("pop_dynamical_python.out", "w+");
	for(i=0; i<100000; i++)
	{	
		for(j=0; j<10; j++)
		{
			fprintf(f, "%lf ", gsl_matrix_get(pop_t, j, i));
		}
		fprintf(f, "\n");
	}
	fclose(f);

	FILE *fini = fopen ("field_ini_python.out", "w+");
	for(i=0; i<250; i++)
	{	
		fprintf(fini, "%lf ", gsl_vector_get(field_ini, i));
	}
	fclose(fini);

	FILE *ffin = fopen ("field_fin_python.out", "w+");
	for(i=0; i<250; i++)
	{	
		fprintf(ffin, "%lf ", gsl_vector_get(field_fin, i));
	}
	fclose(ffin);
}
