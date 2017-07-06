/************************************************************************************/
/************************************************************************************/
//										     /
//				Command to execute code:                             /
//				========================                             /
//	gcc -O3 -Wall $(gsl-config --cflags) <filename> $(gsl-config --libs)  	     /
//										     /
/************************************************************************************/
/************************************************************************************/

/************************************************************************************/


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
#include <gsl/gsl_math.h>			// COMPLEX NUMBERS LIBRARY
#include <gsl/gsl_eigen.h>				// GSL EIGENSYSTEMS LIBRARY
#include <gsl/gsl_linalg.h>				// GSL LINEAR ALGEBRA LIBRARY
#include <gsl/gsl_rng.h>				// GSL RANDOM NUMBER GENERATOR


int print_matrix(gsl_matrix *m)
{
        int status, n = 0;

        for (size_t i = 0; i < m->size1; i++) {
                for (size_t j = 0; j < m->size2; j++) {
                        if ((status = printf("%g ", gsl_matrix_get(m, i, j))) < 0)
                                return -1;
                        n += status;
                }

                if ((status = printf("\n")) < 0)
                        return -1;
                n += status;
        }

        return n;
}

int main()
{

	gsl_matrix *M1 = gsl_matrix_calloc(10, 10);
	gsl_matrix *M1read = gsl_matrix_alloc(10, 10);
	gsl_matrix *M2 = gsl_matrix_calloc(10, 10);
	gsl_matrix *M2read = gsl_matrix_alloc(10, 10);
	
	gsl_matrix_set(M1, 2, 3, 2.0);
	gsl_matrix_set(M2, 3, 7, 1.0);
	
	FILE * file = fopen ("test.dat", "wb");
     	gsl_matrix_fwrite (file, M1);
	gsl_matrix_fwrite (file, M2);
     	fclose (file);

	FILE *readfile = fopen("test.dat","rb");
	gsl_matrix_fread (readfile, M1read);
	gsl_matrix_fread (readfile, M2read);
	fclose(readfile);

	print_matrix(M1read);
	printf("\n");
	print_matrix(M2read);
	
	return GSL_SUCCESS;
}


