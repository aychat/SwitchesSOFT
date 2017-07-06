/************************************************************************************/
/************************************************************************************/
//										     /
//				Command to execute code:                             /
//				========================                             /
//	gcc -O3 -Wall $(gsl-config --cflags) GradientDescent.c $(gsl-config --libs)  /
//										     /
/************************************************************************************/
/************************************************************************************/


/************************************************************************************/
/************************************************************************************/
//										     /
//	A GRADIENT-BASED OPTIMIZATION ALGORITHM TO CALCULATE THE POPULATION	     /
//    			 DYNAMICS FOR AN OPTOGENETIC SWITCH			     /
//										     /
//	     THE COMPLETE DOCUMENTATION IS AVAILABLE IN THE PDF FILE:<Report>        /
//										     /
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
#include <gsl/gsl_complex_math.h>			// COMPLEX NUMBERS LIBRARY
#include <gsl/gsl_eigen.h>				// GSL EIGENSYSTEMS LIBRARY
#include <gsl/gsl_linalg.h>				// GSL LINEAR ALGEBRA LIBRARY
#include <gsl/gsl_rng.h>				// GSL RANDOM NUMBER GENERATOR


typedef struct{
	gsl_matrix_complex *rho;
	gsl_matrix_complex *rho_init;
	gsl_matrix_complex *mu;
	gsl_matrix_complex *mu_init;
	gsl_matrix_complex *Htot;
	gsl_matrix_complex *H0;
	gsl_matrix_complex *g;

	gsl_matrix_complex *mu_rho;
	gsl_matrix_complex *rho_mu;

	gsl_matrix_complex *H_q;
	gsl_matrix_complex *q_H;
	gsl_matrix_complex *Lmat;
	
	const gsl_vector *field;
	gsl_vector *field_plus;
	gsl_vector *field_minus;

	gsl_vector *time;
	double dt;
	double time_initial;
	double time_final;
	int Tratio;
	int time_zero_field_steps;
	double sigma2;
	gsl_vector *frequency;

	gsl_vector *dJ_dE;
	const gsl_vector *A_phi_params;
	gsl_vector *dJ_dA_dphi;

	int j_indx;
	int count;

	double gamma_4_1;
	double gamma_2_3;
	double gamma_3_5;
	double gamma_3_4;
	double gamma_5_6;

	double gamma_9_6;
	double gamma_7_8;
	double gamma_8_9;
	double gamma_8_10;
	double gamma_10_1;
	double gamma_dephasing;

	double freq_central;
	int time_delay;

	gsl_matrix_complex *env_term_4_1;
	gsl_matrix_complex *env_term_2_3;
	gsl_matrix_complex *env_term_3_5;
	gsl_matrix_complex *env_term_3_4;
	gsl_matrix_complex *env_term_5_6;

	gsl_matrix_complex *env_term_9_6;
	gsl_matrix_complex *env_term_7_8;
	gsl_matrix_complex *env_term_8_10;
	gsl_matrix_complex *env_term_8_9;
	gsl_matrix_complex *env_term_10_1;

	gsl_matrix_complex *dephase_term;

	gsl_matrix *pop_t;
	
}parameters;



//***************************************************************************************//
//											 //
//			PRINT FUNCTIONS USED IN THE PROGRAM				 //
//											 //
//***************************************************************************************//


int print_cmplx_mat(gsl_matrix_complex *A)
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
			printf("(%3.3e, %3.3e) ", x, y);
		}
	printf("\n");
	}
	printf("\n\n");
	return GSL_SUCCESS;
}

int print_mat(gsl_matrix *A)
{
	int i,j;
	for(i=0; i<A->size1; i++)
	{
		for(j=0; j<A->size2; j++)
		{	
			printf("%3.3e ", gsl_matrix_get(A,i,j));
		}
	printf("\n");
	}
	printf("\n\n");
	return GSL_SUCCESS;
}
int print_vec(const gsl_vector *A)
{
	const int vec_size = A->size;
	int i;
	for(i=0; i<vec_size; i++)
	{
	printf("%.8e \n", gsl_vector_get(A,i) );
	}
	printf("\n\n");
	
	return GSL_SUCCESS;
}

int trace_cmplx_mat(gsl_matrix_complex *A)
{
	gsl_complex trace = gsl_complex_rect(0.0, 0.0);
	int i;
	const int N = A->size1;
	for(i=0; i<N; i++)
	{
	trace = gsl_complex_add(trace, gsl_matrix_complex_get(A, i, i));
	}

	printf("%3.2f, %3.2f \n \n", GSL_REAL(trace), GSL_IMAG(trace));	

	return GSL_SUCCESS;
}

double max_element(const gsl_matrix_complex *A)
{
	const int N = A->size1;
	int i, j;
	double max_value = gsl_complex_abs(gsl_matrix_complex_get(A, 0, 0));

	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			const double x = gsl_complex_abs(gsl_matrix_complex_get(A, i, j));
			if(x > max_value)
			{	
				max_value = x;
			}
		}
	}
	return max_value;
}


//***************************************************************************************//
//											 //
//			PROPAGATOR FUNCTIONS USED IN THE PROGRAM			 //
//											 //
//***************************************************************************************//

gsl_vector* field_func(const gsl_vector *A_phi_params, parameters *params)
{
	int i,j;
	int Tsteps = params->time->size;
	int Nfreq = params->frequency->size;
	double delay = params->time_delay;
	// double sigma2 = params->sigma2;
	gsl_vector *field = gsl_vector_calloc(Tsteps);

	for(i = 0; i<Tsteps; i++)
	{
		double result = 0.0;
		double t_i = gsl_vector_get(params->time, i);
		double gaussian = gsl_sf_exp(-(t_i-delay)*(t_i-delay)/(2000.));
		// printf("%d %3.4f \n", i, gaussian);

		for (j = 0; j < Nfreq; j++) 
		{
			double x = gsl_vector_get(A_phi_params, j)*gsl_sf_cos(gsl_vector_get(params->frequency, j)*t_i);
			result += x;
    		}

	    	gsl_vector_set(field, i, result*gaussian);
		
	}

	FILE *output_field = fopen ("field.out", "wb");
	gsl_vector_fwrite (output_field, field);
	fclose (output_field);

	return field;
}

int environment_term(gsl_matrix_complex *qmat, const int l, const int m, 
		 const double gamma, gsl_matrix_complex *env_term, parameters *params)
//----------------------------------------------------------------------------------//
// GIVES 2nd TERM IN THE LINDBLAD EQUATION FOR DISSIPATION BETWEEN TWO GIVEN LEVELS //
//----------------------------------------------------------------------------------//
{
	const int N = qmat->size1;
	int i;

	for(i=0; i<N; i++)
	{
		gsl_matrix_complex_set(env_term, m, i, gsl_matrix_complex_get(qmat, m, i));
		gsl_matrix_complex_set(env_term, i, m, gsl_matrix_complex_get(qmat, i, m));
	}
	gsl_matrix_complex_scale(env_term, gsl_complex_rect(-0.50, 0.0));

	gsl_matrix_complex_set(env_term, l, l, gsl_matrix_complex_get(qmat, m, m));	
	gsl_matrix_complex_set(env_term, m, m, gsl_complex_mul(gsl_matrix_complex_get(qmat, m, m), gsl_complex_rect(-1.0, 0.0)));	

	gsl_matrix_complex_scale(env_term, gsl_complex_rect(gamma, 0.0));

	return GSL_SUCCESS;
}

int dephasing_term(gsl_matrix_complex *qmat, gsl_matrix_complex *dephase_term, parameters *params)
//----------------------------------------------------------------------------------//
// 			GIVES Dephasing TERM IN THE LINDBLAD EQUATION	  	    //
//----------------------------------------------------------------------------------//
{
	const int N = qmat->size1;
	int i;

	const double gamma_dephasing = params->gamma_dephasing;
	gsl_matrix_complex_memcpy(dephase_term, qmat);	

	for(i=0; i<N; i++)
	{
		gsl_matrix_complex_set(dephase_term, i, i, gsl_complex_rect(0.0, 0.0));
	}
	gsl_matrix_complex_scale(dephase_term, gsl_complex_rect(-gamma_dephasing, 0.0));

	return GSL_SUCCESS;
}
int L_func(gsl_matrix_complex *qmat, parameters *params)
//----------------------------------------------------//
// 	RETURNS q <-- L[q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
	const double gamma_4_1  = params->gamma_4_1;
	const double gamma_2_3  = params->gamma_2_3;
	const double gamma_3_5  = params->gamma_3_5;
	const double gamma_3_4 = params->gamma_3_4;
	const double gamma_5_6  = params->gamma_5_6;

	const double gamma_9_6 = params->gamma_9_6;
	const double gamma_7_8 = params->gamma_7_8;
	const double gamma_8_10 = params->gamma_8_10;
	const double gamma_8_9 = params->gamma_8_9;
	const double gamma_10_1 = params->gamma_10_1;

	environment_term(qmat, 0, 3, gamma_4_1, params->env_term_4_1, params); 
	environment_term(qmat, 2, 1, gamma_2_3, params->env_term_2_3, params);
	environment_term(qmat, 4, 2, gamma_3_5, params->env_term_3_5, params); 
	environment_term(qmat, 3, 2, gamma_3_4, params->env_term_3_4, params);
	environment_term(qmat, 5, 4, gamma_5_6, params->env_term_5_6, params); 

	environment_term(qmat, 5, 8, gamma_9_6, params->env_term_9_6, params);
	environment_term(qmat, 7, 6, gamma_7_8, params->env_term_7_8, params);
	environment_term(qmat, 9, 7, gamma_8_10, params->env_term_8_10, params);
	environment_term(qmat, 8, 7, gamma_8_9, params->env_term_8_9, params);
	environment_term(qmat, 0, 9, gamma_10_1, params->env_term_10_1, params);
	dephasing_term(qmat, params->dephase_term, params);

	const gsl_complex set_one = gsl_complex_rect(1.0, 0.0);
	const gsl_complex set_zero = gsl_complex_rect(0.0, 0.0);

	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, params->Htot, qmat, set_zero, params->H_q);
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, qmat, params->Htot, set_zero, params->q_H);

	gsl_matrix_complex_sub(params->H_q, params->q_H);
	gsl_matrix_complex_scale(params->H_q, gsl_complex_rect(0.0, -1.0));

	gsl_matrix_complex_add(params->H_q, params->env_term_4_1);
	gsl_matrix_complex_add(params->H_q, params->env_term_2_3);
	gsl_matrix_complex_add(params->H_q, params->env_term_3_5);
	gsl_matrix_complex_add(params->H_q, params->env_term_3_4);
	gsl_matrix_complex_add(params->H_q, params->env_term_5_6);

	gsl_matrix_complex_add(params->H_q, params->env_term_9_6);
	gsl_matrix_complex_add(params->H_q, params->env_term_7_8);
	gsl_matrix_complex_add(params->H_q, params->env_term_8_10);
	gsl_matrix_complex_add(params->H_q, params->env_term_8_9);
	gsl_matrix_complex_add(params->H_q, params->env_term_10_1);

	gsl_matrix_complex_add(params->H_q, params->dephase_term);

	gsl_matrix_complex_memcpy(params->Lmat, params->H_q);
	return GSL_SUCCESS;
}

int propagate(gsl_matrix_complex *qmat, double field, parameters *params)
//------------------------------------------------------------//
//     RETURNS q <-- e^{(dt)L[q]} AT A PARTICULAR TIME (t)    //
//------------------------------------------------------------//
{
	const double dt = params->dt;
	gsl_matrix_complex_memcpy(params->Lmat, qmat);
	int k = 1;
	double max_el;

	gsl_matrix_complex_memcpy(params->mu, params->mu_init);
	gsl_matrix_complex_scale(params->mu, gsl_complex_rect(field, 0.0));
	gsl_matrix_complex_memcpy(params->Htot, params->H0);
	gsl_matrix_complex_add(params->Htot, params->mu);

	do											// COMPUTES e^{(dt)L[q(t)]} 	//
	{
		L_func(params->Lmat, params);							// L[.] <--- L[L[.]]		//
		gsl_matrix_complex_scale(params->Lmat, gsl_complex_rect(dt/k, 0.0));		// L[.] <--- L[.]*(dt)/k  	//
		gsl_matrix_complex_add(qmat, params->Lmat);					// q(t) <--- q(t) + L[.]	//
		max_el = max_element(params->Lmat);
		k += 1;

	}while(max_el > 1e-8);

	return GSL_SUCCESS;
}

int g_propagate_t_T(int tau_indx, const gsl_vector *field, parameters *params)
//------------------------------------------------------------//
//     RETURNS g(t,T) from g(t,t) AT A PARTICULAR TIME (t)    //
//------------------------------------------------------------//
{
	int i;

	const gsl_complex set_zero = gsl_complex_rect(0.0, 0.0);
	const gsl_complex set_one = gsl_complex_rect(1.0, 0.0);

	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, params->mu_init, params->rho, set_zero, params->mu_rho);
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, params->rho, params->mu_init, set_zero, params->rho_mu);

	gsl_matrix_complex_sub(params->mu_rho, params->rho_mu);
	gsl_matrix_complex_scale(params->mu_rho, gsl_complex_rect(0.0, -1.0));

	gsl_matrix_complex_memcpy(params->g, params->mu_rho);

	for(i=tau_indx; i<params->time->size; i++)
	{
		propagate(params->g, gsl_vector_get(field, i), params);
	}

	return GSL_SUCCESS;
}

int rho_propagate_0_T(const gsl_vector *field, parameters *params)
//------------------------------------------------------------//
//    GETTING rho(T) FROM rho(0) USING PROPAGATE FUNCTION     //
//------------------------------------------------------------//
{

	int i,j;

	gsl_matrix_complex_memcpy(params->rho, params->rho_init);

	for(i=0; i<params->time->size; i++)
	{	
		propagate(params->rho, gsl_vector_get(field, i), params);
		for(j=0; j<params->rho->size1; j++)
		{
			gsl_matrix_set(params->pop_t, j, i, GSL_REAL(gsl_matrix_complex_get(params->rho, j, j)));
		}

	}
	
	return GSL_SUCCESS;
}

int propagate_zero_field(gsl_matrix_complex *qmat, parameters *params)
//------------------------------------------------------------//
//     RETURNS q <-- e^{(dt)L[q]} AT A PARTICULAR TIME (t)    //
//------------------------------------------------------------//
{
	const double dt = params->dt;
	gsl_matrix_complex_memcpy(params->Lmat, qmat);
	int k = 1;
	double max_el;

	do											// COMPUTES e^{(dt)L[q(t)]} 	//
	{
		L_func(params->Lmat, params);							// L[.] <--- L[L[.]]		//
		gsl_matrix_complex_scale(params->Lmat, gsl_complex_rect(dt/k, 0.0));		// L[.] <--- L[.]*(dt)/k  	//
		gsl_matrix_complex_add(qmat, params->Lmat);					// q(t) <--- q(t) + L[.]	//
		max_el = max_element(params->Lmat);
		k += 1;

	}while(max_el > 1e-8);

	return GSL_SUCCESS;
}

int rho_propagate_T_T_total(parameters *params)
//------------------------------------------------------------//
//    GETTING rho(T) FROM rho(0) USING PROPAGATE FUNCTION     //
//------------------------------------------------------------//
{

	int i, j;
	gsl_matrix_complex_memcpy(params->Htot, params->H0);
	const int time_size = params->time->size;

	for(i = time_size; i < time_size + params->time_zero_field_steps; i++)
	{	
		propagate_zero_field(params->rho, params);
		for(j=0; j<params->rho->size1; j++)
		{
			gsl_matrix_set(params->pop_t, j, i, GSL_REAL(gsl_matrix_complex_get(params->rho, j, j)));
		}
	}
	
	return GSL_SUCCESS;
}
int calculate_dJ_dE(const gsl_vector *A_phi_params, parameters *params)
{
	const int Tsteps = params->time->size;
	const int Nfreq = params->frequency->size;
	int  tau_indx, j;
	const double dt = params->dt;
	
	gsl_matrix_complex_memcpy(params->rho, params->rho_init);

	for(tau_indx=0;  tau_indx<Tsteps;  tau_indx++)
	{
		g_propagate_t_T(tau_indx, field_func(A_phi_params, params), params);
		gsl_vector_set(params->dJ_dE, tau_indx, GSL_REAL(gsl_matrix_complex_get(params->g, params->j_indx, params->j_indx)));
		propagate(params->rho, gsl_vector_get(field_func(A_phi_params, params), tau_indx), params);
	}

	for(j=0; j<Nfreq; j++)
	{
		double dA = 0.0;
		double dphi = 0.0;
		int k;
		for(tau_indx=0;  tau_indx<Tsteps;  tau_indx++)
		{
			double gaussian = gsl_sf_exp(-tau_indx*dt*tau_indx*dt/(2.0*params->sigma2));
			double x = gsl_sf_cos(gsl_vector_get(params->frequency, j)*tau_indx*dt + gsl_vector_get(A_phi_params, j+Nfreq));

			double y = -gsl_vector_get(A_phi_params, j)
				*gsl_sf_sin(gsl_vector_get(params->frequency, j)*tau_indx*dt + gsl_vector_get(A_phi_params, j+Nfreq));

			if(tau_indx == 0) k=1;
			else if(tau_indx == (Tsteps-1)) k=1;
			else if( tau_indx % 2 == 0) k=2;
			else if( tau_indx % 2 == 1) k=4;

			dA += gsl_vector_get(params->dJ_dE,tau_indx)*gaussian*x*k;
			dphi += gsl_vector_get(params->dJ_dE,tau_indx)*gaussian*y*k;
		}
		gsl_vector_set(params->dJ_dA_dphi, j, dA*dt/3.0);
		gsl_vector_set(params->dJ_dA_dphi, j+Nfreq, dphi*dt/3.0);
	}


	return GSL_SUCCESS;
}

int Evolution(double t, const double E[], double dE_ds[], void *params)
{
	parameters *Ev_params = (parameters *) params;

	(Ev_params->count)++;
	printf("Adaptive step = %d \n", Ev_params->count);

	gsl_vector_const_view tmp1 = gsl_vector_const_view_array(E, 2*Ev_params->frequency->size);
	gsl_vector_view tmp2 = gsl_vector_view_array(dE_ds, 2*Ev_params->frequency->size);

	Ev_params->A_phi_params = &tmp1.vector;
	Ev_params->dJ_dA_dphi = &tmp2.vector;
	
	calculate_dJ_dE(Ev_params->A_phi_params, params);

	return GSL_SUCCESS;
}

int main(int argc, char *argv[])
{

	const int N = 10;
	int Nfreq = 4;
	int i;
	const int j_indx = 1;

	double freq_central = .4796679; 				// in fs-1

	double gamma_4_1  = 1./(freq_central*150.);			// 150 fs
	double gamma_2_3  = 1./(freq_central*150.);			// 150 ps
	double gamma_3_5  = 1./(freq_central*61.*1000.);		// 61. ps
	double gamma_3_4  = 1./(freq_central*26.*1000.);		// 26. ps
	double gamma_5_6  = 1./(freq_central*1.);			// 1 ms

	double gamma_9_6  = 1./(freq_central*10.);			// 10 fs
	double gamma_7_8  = 1./(freq_central*50.);    	 		// 50 fs
	double gamma_8_10 = 1./(freq_central*1.5*10000.);		// 1.5 ps
	double gamma_8_9  = 1./(freq_central*300.);			// 300 fs
	double gamma_10_1 = 1./(freq_central*1.);			// 1 ms

	double gamma_dephasing = 1./(freq_central*300.);			// 30 fs

//	const double field_strength = atof (argv[1]);

	const int Tsteps = 250;

	const double dt = 1.;
	const int Tsteps_zero_field = 100000-Tsteps;
	const double time_initial = 0.00;
	const double time_final = dt*Tsteps;
	const int time_delay = (time_final-time_initial)/1.8;
	const double sigma2 = (time_final-time_initial)/0.20;	// in fs from TBWP relation

	gsl_vector *time = gsl_vector_alloc(Tsteps);
	for(i = 0; i<Tsteps; i++)
	{
		gsl_vector_set(time, i, time_initial+dt*i);
	}

	gsl_vector *A_phi_params = gsl_vector_alloc(2*Nfreq);

/*
	for(i=0; i<Nfreq; i++)
	{
		gsl_vector_set(A_phi_params, i, field_strength);
		gsl_vector_set(A_phi_params, i+Nfreq, 0.00);
	}
*/

	FILE *field_f = fopen ("field_out_params.out", "rb");
	gsl_vector_fread(field_f, A_phi_params);
	fclose (field_f);

	double E[10];
	E[0] = 0.00;
	E[1] = 1.00;
	E[2] = 0.98;
	E[3] = 0.02;
	E[4] = 0.50;
	E[5] = 0.01;
	E[6] = 0.75;
	E[7] = 0.72;
	E[8] = 0.04;
	E[9] = 0.48;

	gsl_vector *frequency = gsl_vector_alloc(Nfreq);

	gsl_vector_set(frequency, 0, E[1]-E[0]);
	gsl_vector_set(frequency, 1, E[2]-E[3]);
	gsl_vector_set(frequency, 2, E[6]-E[5]);
	gsl_vector_set(frequency, 3, E[7]-E[8]);

	gsl_matrix_complex *H0 = gsl_matrix_complex_calloc(N, N);
	gsl_matrix_complex_set(H0, 0, 0, gsl_complex_rect(E[0], 0.0));
	gsl_matrix_complex_set(H0, 1, 1, gsl_complex_rect(E[1], 0.0));
	gsl_matrix_complex_set(H0, 2, 2, gsl_complex_rect(E[2], 0.0));
	gsl_matrix_complex_set(H0, 3, 3, gsl_complex_rect(E[3], 0.0));
	gsl_matrix_complex_set(H0, 4, 4, gsl_complex_rect(E[4], 0.0));
	gsl_matrix_complex_set(H0, 5, 5, gsl_complex_rect(E[5], 0.0));
	gsl_matrix_complex_set(H0, 6, 6, gsl_complex_rect(E[6], 0.0));
	gsl_matrix_complex_set(H0, 7, 7, gsl_complex_rect(E[7], 0.0));
	gsl_matrix_complex_set(H0, 8, 8, gsl_complex_rect(E[8], 0.0));
	gsl_matrix_complex_set(H0, 9, 9, gsl_complex_rect(E[9], 0.0));

	gsl_matrix_complex *mu_init = gsl_matrix_complex_calloc(N, N);
	gsl_matrix_complex_set(mu_init, 0, 1, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 1, 0, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 2, 3, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 3, 2, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 5, 6, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 6, 5, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 7, 8, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 8, 7, gsl_complex_rect(1.0, 0.0));

	gsl_matrix_complex *rho_init = gsl_matrix_complex_calloc(N, N);

	parameters params;
	params.freq_central = freq_central;
	params.time_delay = time_delay;
	params.rho = gsl_matrix_complex_calloc(N, N);
	params.rho_init = rho_init;
	params.Htot = gsl_matrix_complex_calloc(N, N);
	params.H0 = H0;
	params.g = gsl_matrix_complex_calloc(N, N);	
	params.mu = gsl_matrix_complex_calloc(N, N);
	params.mu_init = mu_init;
	params.mu_rho = gsl_matrix_complex_calloc(N, N);
	params.rho_mu = gsl_matrix_complex_calloc(N, N);
	params.H_q = gsl_matrix_complex_calloc(N, N);
	params.q_H = gsl_matrix_complex_calloc(N, N);
	params.Lmat = gsl_matrix_complex_calloc(N, N);
	params.dt = dt;
	params.time_final = time_final;
	params.time = time;
	params.time_zero_field_steps = Tsteps_zero_field;
	params.A_phi_params = gsl_vector_alloc(2*Nfreq);
	params.field_plus = gsl_vector_alloc(Tsteps);
	params.field_minus = gsl_vector_alloc(Tsteps);
	params.dJ_dE = gsl_vector_alloc(Tsteps);
	params.j_indx = j_indx;
	params.count = 0;
	params.sigma2 = sigma2;
	params.dJ_dA_dphi = gsl_vector_alloc(2*Nfreq);
	params.frequency = frequency;

	params.gamma_4_1 = gamma_4_1;
	params.gamma_2_3 = gamma_2_3;
	params.gamma_3_5 = gamma_3_5;
	params.gamma_3_4 = gamma_3_4;
	params.gamma_5_6 = gamma_5_6;

	params.gamma_9_6 = gamma_9_6;
	params.gamma_7_8 = gamma_7_8;
	params.gamma_8_9 = gamma_8_9;
	params.gamma_8_10 = gamma_8_10;
	params.gamma_10_1 = gamma_10_1;

	params.gamma_dephasing = gamma_dephasing;

	params.pop_t = gsl_matrix_alloc(N, Tsteps+Tsteps_zero_field);

	params.env_term_4_1  = gsl_matrix_complex_alloc(N, N);
	params.env_term_2_3  = gsl_matrix_complex_alloc(N, N);
	params.env_term_3_5  = gsl_matrix_complex_alloc(N, N);
	params.env_term_3_4  = gsl_matrix_complex_alloc(N, N);
	params.env_term_5_6  = gsl_matrix_complex_alloc(N, N);

	params.env_term_9_6  = gsl_matrix_complex_alloc(N, N);
	params.env_term_7_8  = gsl_matrix_complex_alloc(N, N);
	params.env_term_8_10 = gsl_matrix_complex_alloc(N, N);
	params.env_term_8_9  = gsl_matrix_complex_alloc(N, N);
	params.env_term_10_1 = gsl_matrix_complex_alloc(N, N);

	params.dephase_term = gsl_matrix_complex_alloc(N, N);

/*	gsl_matrix_complex_set(rho_init, 2, 2, gsl_complex_rect(1.0, 0.0));	
	gsl_matrix_complex_memcpy(params.rho, params.rho_init);

	rho_propagate_0_T(field_func(A_phi_params, &params), &params);
	rho_propagate_T_T_total(&params);
*/
	int j;
	// FILE *pop_file = fopen ("pop_M.out", "w+");
	for(j = 0; j< 1; j++)
	{
		printf("%d \n", j+1);
		gsl_matrix_complex_set_all(rho_init, gsl_complex_rect(0.0, 0.0));
		gsl_matrix_complex_set(rho_init, j, j, gsl_complex_rect(0.5, 0.0));
		gsl_matrix_complex_set(rho_init, j+5, j+5, gsl_complex_rect(0.5, 0.0));	
		gsl_matrix_complex_memcpy(params.rho, params.rho_init);

		rho_propagate_0_T(field_func(A_phi_params, &params), &params);
		rho_propagate_T_T_total(&params);
		for(i = 0; i< N; i++)
		{
			printf("%3.6f       %d  \n", GSL_REAL(gsl_matrix_complex_get(params.rho, i, i)), i+1);
		}

		FILE *output_file = fopen ("pop_dynamical.out", "wb");
		gsl_matrix_fwrite (output_file, params.pop_t);
		fclose (output_file);
/*
		for(i = 0; i< N; i++)
		{
			fprintf(pop_file, "%lf ", gsl_matrix_get(params.pop_t, i, 99999));
		}
		fprintf(pop_file, "\n"); */ 
	}
	// fclose (pop_file);

	return GSL_SUCCESS;
}

