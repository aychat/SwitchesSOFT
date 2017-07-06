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

	double gamma_7_6;
	double gamma_6_5;
	double gamma_5_4;
	double gamma_4_3;
	double gamma_3_2;
	double gamma_2_1;
	double gamma_1_0;
	double gamma_dephasing;

	double freq_central;
	int time_delay;

	gsl_matrix_complex *env_term_7_6;
	gsl_matrix_complex *env_term_6_5;
	gsl_matrix_complex *env_term_5_4;
	gsl_matrix_complex *env_term_4_3;
	gsl_matrix_complex *env_term_3_2;
	gsl_matrix_complex *env_term_2_1;
	gsl_matrix_complex *env_term_1_0;

	gsl_matrix_complex *dephase_term;

	gsl_matrix_complex *A_mat_7_6;
	gsl_matrix_complex *A_mat_6_5;
	gsl_matrix_complex *A_mat_5_4;
	gsl_matrix_complex *A_mat_4_3;
	gsl_matrix_complex *A_mat_3_2;
	gsl_matrix_complex *A_mat_2_1;
	gsl_matrix_complex *A_mat_1_0;

	gsl_matrix_complex *A_Adag;
	gsl_matrix_complex *A_A_rho;
	gsl_matrix_complex *rho_A_A;
	
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
			// printf("(%3.2lf, %3.2lf) ", x, y);
			printf("%3.2lf ", x);
		}
	printf("\n");
	}
	printf("\n\n");
	return GSL_SUCCESS;
}

int print_cmplx_vec(gsl_vector_complex *A)
{
	int vec_size = A->size;
	int i;
	double x, y;
	for(i=0; i<vec_size; i++)
	{
		x = GSL_REAL(gsl_vector_complex_get(A,i));
		y = GSL_IMAG(gsl_vector_complex_get(A,i));
		// printf("(%3.3e, %3.3e) ", x, y);
		printf("%3.2lf ", x);
	}
	printf("\n");
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

	printf("%3.8f, %3.8f \n \n", GSL_REAL(trace), GSL_IMAG(trace));	

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
	double sigma2 = params->sigma2;
	gsl_vector *field = gsl_vector_calloc(Tsteps);

	for(i = 0; i<Tsteps; i++)
	{
		double result = 0.0;
		double t_i = gsl_vector_get(params->time, i);
		double gaussian = gsl_sf_exp(-(t_i-delay)*(t_i-delay)/sigma2);

		for (j = 0; j < Nfreq; j++) 
		{
			double x = gsl_vector_get(A_phi_params, j)*gsl_sf_cos(gsl_vector_get(params->frequency, j)*t_i);
			result += x;
    		}

	    	gsl_vector_set(field, i, result*gaussian);
	}

	gsl_vector_scale(field, 1.0/params->freq_central);

	return field;
}

int environment_term(gsl_matrix_complex *qmat, gsl_matrix_complex *Amat, 
		 const double gamma, gsl_matrix_complex *env_term, parameters *params)
//----------------------------------------------------------------------------------//
// GIVES 2nd TERM IN THE LINDBLAD EQUATION FOR DISSIPATION BETWEEN TWO GIVEN LEVELS //
//----------------------------------------------------------------------------------//
{
	const gsl_complex set_one = gsl_complex_rect(1.0, 0.0);
	const gsl_complex set_zero = gsl_complex_rect(0.0, 0.0);
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, Amat, qmat, set_zero, env_term);
	gsl_blas_zgemm(CblasNoTrans, CblasTrans, set_one, env_term, Amat, set_zero, env_term);

	gsl_blas_zgemm(CblasTrans, CblasNoTrans, set_one, Amat, Amat, set_zero, params->A_Adag);
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, params->A_Adag, qmat, set_zero, params->A_A_rho);
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, qmat, params->A_Adag, set_one, params->A_A_rho);

	// gsl_matrix_complex_add(params->A_A_rho, params->rho_A_A);
	gsl_matrix_complex_scale(params->A_A_rho, gsl_complex_rect(-0.5, 0.0));

	gsl_matrix_complex_add(env_term, params->A_A_rho);

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
	environment_term(qmat, params->A_mat_7_6, params->gamma_7_6, params->env_term_7_6, params); 
	environment_term(qmat, params->A_mat_6_5, params->gamma_6_5, params->env_term_6_5, params);
	environment_term(qmat, params->A_mat_5_4, params->gamma_5_4, params->env_term_5_4, params); 
	environment_term(qmat, params->A_mat_4_3, params->gamma_4_3, params->env_term_4_3, params);
	environment_term(qmat, params->A_mat_3_2, params->gamma_3_2, params->env_term_3_2, params); 
	environment_term(qmat, params->A_mat_2_1, params->gamma_2_1, params->env_term_2_1, params);
	environment_term(qmat, params->A_mat_1_0, params->gamma_1_0, params->env_term_1_0, params);

	dephasing_term(qmat, params->dephase_term, params);

	const gsl_complex set_one = gsl_complex_rect(1.0, 0.0);
	const gsl_complex set_zero = gsl_complex_rect(0.0, 0.0);

	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, params->Htot, qmat, set_zero, params->H_q);
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, qmat, params->Htot, set_zero, params->q_H);

	gsl_matrix_complex_sub(params->H_q, params->q_H);
	gsl_matrix_complex_scale(params->H_q, gsl_complex_rect(0.0, -1.0));

	// gsl_matrix_complex_add(params->H_q, params->env_term_7_6);
	// gsl_matrix_complex_add(params->H_q, params->env_term_6_5);
	// gsl_matrix_complex_add(params->H_q, params->env_term_5_4);
	// gsl_matrix_complex_add(params->H_q, params->env_term_4_3);
	// gsl_matrix_complex_add(params->H_q, params->env_term_3_2);
	// gsl_matrix_complex_add(params->H_q, params->env_term_2_1);
	// gsl_matrix_complex_add(params->H_q, params->env_term_1_0);

	// gsl_matrix_complex_add(params->H_q, params->dephase_term);

	gsl_matrix_complex_memcpy(params->Lmat, params->H_q);

	// trace_cmplx_mat(params->Lmat);
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

	const int N = 8;
	int Nfreq = 1;
	int i;
	double PRsum;
	const int j_indx = 1;

	double freq_central = .454; 					// in fs-1

	double gamma_7_6  = 1./(freq_central*50.);			// 50 fs
	double gamma_6_5  = 1./(freq_central*50.);			// 50 fs
	double gamma_5_4  = 1./(freq_central*50.);			// 50 fs

	double gamma_4_3  = 1./(freq_central*26.*1000.);		// 26 ps

	double gamma_3_2  = 1./(freq_central*50.);			// 50 fs
	double gamma_2_1  = 1./(freq_central*50.);			// 50 fs
	double gamma_1_0  = 1./(freq_central*50.);    	 		// 50 fs

 	const double field_strength = atof(argv[1]);
	const double dephasing_g = atof(argv[2]);

	double gamma_dephasing = 1./(freq_central*dephasing_g);		// in fs

	const int Tsteps = atoi(argv[3]);
	const int n_lvl = 4;
	const double dt = 1.0;
	const int Tsteps_zero_field = 0;
	const double time_initial = 0.00;
	const double time_final = dt*Tsteps;
	const int time_delay = (time_final-time_initial)/2.;
	const double sigma2 = (time_final-time_initial)/0.2;	

	gsl_vector *time = gsl_vector_alloc(Tsteps);
	for(i = 0; i<Tsteps; i++)
	{
		gsl_vector_set(time, i, time_initial+dt*i);
	}

	gsl_vector *A_phi_params = gsl_vector_alloc(2*Nfreq);

	gsl_vector_set(A_phi_params, 0, field_strength);
	gsl_vector_set(A_phi_params, 1, 0.00);

	double E[N];
	double omega = .1075;
	double gap = 1.0;
	E[0] = 0.5*omega;
	E[1] = 1.5*omega;
	E[2] = 2.5*omega;
	E[3] = 3.5*omega;

	E[4] = 0.5*omega + gap;
	E[5] = 1.5*omega + gap;
	E[6] = 2.5*omega + gap;
	E[7] = 3.5*omega + gap;

	gsl_vector *frequency = gsl_vector_alloc(Nfreq);

	gsl_matrix_complex *H0 = gsl_matrix_complex_calloc(N, N);
	gsl_matrix_complex *mu_init = gsl_matrix_complex_calloc(N, N);
	gsl_matrix_complex_set_all(mu_init, gsl_complex_rect(1.0, 0.0));
	for (i=0; i<N; i++)
	{
		gsl_matrix_complex_set(H0, i, i, gsl_complex_rect(E[i], 0.0));
		gsl_matrix_complex_set(mu_init, i, i, gsl_complex_rect(0.0, 0.0));
	}
	
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
	params.field = gsl_vector_alloc(Tsteps);
	params.field_plus = gsl_vector_alloc(Tsteps);
	params.field_minus = gsl_vector_alloc(Tsteps);
	params.dJ_dE = gsl_vector_alloc(Tsteps);
	params.j_indx = j_indx;
	params.count = 0;
	params.sigma2 = sigma2;
	params.dJ_dA_dphi = gsl_vector_alloc(2*Nfreq);
	params.frequency = frequency;
	params.A_Adag = gsl_matrix_complex_alloc(N, N);
	params.A_A_rho = gsl_matrix_complex_alloc(N, N);
	params.rho_A_A = gsl_matrix_complex_alloc(N, N);

	params.gamma_7_6 = gamma_7_6;
	params.gamma_6_5 = gamma_6_5;
	params.gamma_5_4 = gamma_5_4;
	params.gamma_4_3 = gamma_4_3;
	params.gamma_3_2 = gamma_3_2;
	params.gamma_2_1 = gamma_2_1;
	params.gamma_1_0 = gamma_1_0;

	params.gamma_dephasing = gamma_dephasing;

	params.pop_t = gsl_matrix_alloc(N, Tsteps+Tsteps_zero_field);

	params.env_term_7_6  = gsl_matrix_complex_alloc(N, N);
	params.env_term_6_5  = gsl_matrix_complex_alloc(N, N);
	params.env_term_5_4  = gsl_matrix_complex_alloc(N, N);
	params.env_term_4_3  = gsl_matrix_complex_alloc(N, N);
	params.env_term_3_2  = gsl_matrix_complex_alloc(N, N);
	params.env_term_2_1  = gsl_matrix_complex_alloc(N, N);
	params.env_term_1_0  = gsl_matrix_complex_alloc(N, N);

	params.A_mat_7_6 = gsl_matrix_complex_alloc(N, N);
	params.A_mat_6_5 = gsl_matrix_complex_alloc(N, N);
	params.A_mat_5_4 = gsl_matrix_complex_alloc(N, N);
	params.A_mat_4_3 = gsl_matrix_complex_alloc(N, N);
	params.A_mat_3_2 = gsl_matrix_complex_alloc(N, N);
	params.A_mat_2_1 = gsl_matrix_complex_alloc(N, N);
	params.A_mat_1_0 = gsl_matrix_complex_alloc(N, N);

	params.dephase_term = gsl_matrix_complex_alloc(N, N);

	FILE *output_field = fopen ("field_ini.txt", "w");
	for(i=0; i<Tsteps; i++)
	{
		fprintf (output_field, "%3.2lf ", gsl_vector_get(field_func(A_phi_params, &params), i));
	}
	fclose (output_field);

	gsl_vector_complex *A_matrix_whole = gsl_vector_complex_alloc((N-1)*N*N);
	
	FILE *A_matrix = fopen("A_matrix.txt", "r");
	int shape = pow((2*n_lvl), 2);
	double var;
	for (i=0; i<(N-1)*N*N; i++)
	{
		fscanf(A_matrix, "%lf ", &var);
		gsl_vector_complex_set(A_matrix_whole, i, gsl_complex_rect(var, 0.0));
	}		
	fclose(A_matrix);

	for (i=0; i<N; i++)
	{
		for(int j=0; j<N; j++)
		{
			gsl_matrix_complex_set(params.A_mat_7_6, i, j, gsl_vector_complex_get(A_matrix_whole, i*N + j));
			gsl_matrix_complex_set(params.A_mat_6_5, i, j, gsl_vector_complex_get(A_matrix_whole, i*N + j + N*N));
			gsl_matrix_complex_set(params.A_mat_5_4, i, j, gsl_vector_complex_get(A_matrix_whole, i*N + j + N*N*2));
			gsl_matrix_complex_set(params.A_mat_4_3, i, j, gsl_vector_complex_get(A_matrix_whole, i*N + j + N*N*3));
			gsl_matrix_complex_set(params.A_mat_3_2, i, j, gsl_vector_complex_get(A_matrix_whole, i*N + j + N*N*4));
			gsl_matrix_complex_set(params.A_mat_2_1, i, j, gsl_vector_complex_get(A_matrix_whole, i*N + j + N*N*5));
			gsl_matrix_complex_set(params.A_mat_1_0, i, j, gsl_vector_complex_get(A_matrix_whole, i*N + j + N*N*6));
		}

	}

	print_cmplx_mat(params.A_mat_7_6);
	print_cmplx_mat(params.A_mat_6_5);
	print_cmplx_mat(params.A_mat_5_4);
	print_cmplx_mat(params.A_mat_4_3);
	print_cmplx_mat(params.A_mat_3_2);
	print_cmplx_mat(params.A_mat_2_1);
	print_cmplx_mat(params.A_mat_1_0);

	FILE *output_file = fopen ("pop_dynamical.out", "wb");

	int j, Nf = 100;
	
	for(j=0; j<5*Nf+1; j++)
	{
		gsl_vector_set(params.frequency, 0, 8.25 + j*1.0/Nf);

		gsl_matrix_complex_set_all(rho_init, gsl_complex_rect(0.0, 0.0));
		gsl_matrix_complex_set(rho_init, 0, 0, gsl_complex_rect(1.0, 0.0));	
		gsl_matrix_complex_memcpy(params.rho, params.rho_init);
		rho_propagate_0_T(field_func(A_phi_params, &params), &params);
		rho_propagate_T_T_total(&params);

		PRsum = 0.0;
		for(i = N/2; i< N; i++)
		{
			PRsum += GSL_REAL(gsl_matrix_complex_get(params.rho, i, i));

		}
		fprintf(output_file, "%3.6lf    %3.6lf \n", gsl_vector_get(params.frequency, 0), PRsum);
		// printf("%3.6lf    %3.6lf \n", gsl_vector_get(params.frequency, 0), PRsum);

	}

	fclose (output_file);


	return GSL_SUCCESS;
}

