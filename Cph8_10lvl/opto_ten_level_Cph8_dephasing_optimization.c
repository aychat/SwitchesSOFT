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
//    			 DYNAMICS FOR A THREE_-LEVEL SYSTEM			     /
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
	double time_central;
	double sigma2;
	gsl_vector *frequency;

	gsl_vector *dJ_dE;
	const gsl_vector *A_phi_params;
	gsl_vector *dJ_dA_dphi;

	int j_indx_max;
	int j_indx_min;
	int count;

	double gamma_23;
	double gamma_34;
	double gamma_35;
	double gamma_41;
	double gamma_78;
	double gamma_89;
	double gamma_810;
	double gamma_96;
	double gamma_101;
	double gamma_56;

	double gamma_dephasing;

	gsl_matrix_complex *env_term_23;
	gsl_matrix_complex *env_term_34;
	gsl_matrix_complex *env_term_35;
	gsl_matrix_complex *env_term_41;
	gsl_matrix_complex *env_term_78;
	gsl_matrix_complex *env_term_89;
	gsl_matrix_complex *env_term_810;
	gsl_matrix_complex *env_term_96;
	gsl_matrix_complex *env_term_101;
	gsl_matrix_complex *env_term_56;

	gsl_matrix_complex *dephase_term;

	gsl_vector *pop_t_1;
	gsl_vector *pop_t_2;
	gsl_vector *pop_t_3;
	gsl_vector *pop_t_4;
	gsl_vector *pop_t_5;
	gsl_vector *pop_t_6;
	gsl_vector *pop_t_7;
	gsl_vector *pop_t_8;
	gsl_vector *pop_t_9;
	gsl_vector *pop_t_10;


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

int print_diag(gsl_matrix_complex *A)
{
	const int N = A->size1;
	int i;
	for(i=0; i<N; i++)
	{
	printf("%5.4f    %d\n", GSL_REAL(gsl_matrix_complex_get(A, i, i)), i+1);
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

	printf("%5.4f, %5.4f \n \n", GSL_REAL(trace), GSL_IMAG(trace));	

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
	gsl_vector *field = gsl_vector_calloc(Tsteps);

	for(i = 0; i<Tsteps; i++)
	{
		double result = 0.0;
		double t_i = gsl_vector_get(params->time, i);
		double gaussian_t = exp(-(t_i-params->time_central)*(t_i-params->time_central)/(2.0*params->sigma2));
		//double sin_env = gsl_sf_pow_int(gsl_sf_sin(2.0*t_i/params->time_final),2);

		for (j = 0; j < Nfreq; j++) 
		{
			double x = gsl_vector_get(A_phi_params, j)*gsl_sf_cos(gsl_vector_get(params->frequency, j)*t_i 
								+ gsl_vector_get(A_phi_params, j+Nfreq));
			result += x;
    		}

	    	//gsl_vector_set(field, i, result*sin_env);
	    	gsl_vector_set(field, i, result*gaussian_t);
		
	}

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
	const double gamma_23 = params->gamma_23;
	const double gamma_34 = params->gamma_34;
	const double gamma_35 = params->gamma_35;
	const double gamma_41 = params->gamma_41;
	const double gamma_78 = params->gamma_78;
	const double gamma_89 = params->gamma_89;
	const double gamma_810 = params->gamma_810;
	const double gamma_96 = params->gamma_96;
	const double gamma_101 = params->gamma_101;
	const double gamma_56 = params->gamma_56;

	environment_term(qmat, 2, 1, gamma_23, params->env_term_23, params); 
	environment_term(qmat, 3, 2, gamma_34, params->env_term_34, params);
	environment_term(qmat, 4, 2, gamma_35, params->env_term_35, params); 
	environment_term(qmat, 0, 3, gamma_41, params->env_term_41, params);
	environment_term(qmat, 7, 6, gamma_78, params->env_term_78, params); 
	environment_term(qmat, 8, 7, gamma_89, params->env_term_89, params);
	environment_term(qmat, 9, 7, gamma_810, params->env_term_810, params); 
	environment_term(qmat, 5, 8, gamma_96, params->env_term_96, params);
	environment_term(qmat, 0, 9, gamma_101, params->env_term_101, params); 
	environment_term(qmat, 5, 4, gamma_56, params->env_term_56, params);

	dephasing_term(qmat, params->dephase_term, params);

	const gsl_complex set_one = gsl_complex_rect(1.0, 0.0);
	const gsl_complex set_zero = gsl_complex_rect(0.0, 0.0);

	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, params->Htot, qmat, set_zero, params->H_q);
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, qmat, params->Htot, set_zero, params->q_H);

	gsl_matrix_complex_sub(params->H_q, params->q_H);
	gsl_matrix_complex_scale(params->H_q, gsl_complex_rect(0.0, -1.0));

	gsl_matrix_complex_add(params->H_q, params->env_term_23);
	gsl_matrix_complex_add(params->H_q, params->env_term_34);
	gsl_matrix_complex_add(params->H_q, params->env_term_35);
	gsl_matrix_complex_add(params->H_q, params->env_term_41);
	gsl_matrix_complex_add(params->H_q, params->env_term_78);
	gsl_matrix_complex_add(params->H_q, params->env_term_89);
	gsl_matrix_complex_add(params->H_q, params->env_term_810);
	gsl_matrix_complex_add(params->H_q, params->env_term_96);
	gsl_matrix_complex_add(params->H_q, params->env_term_101);
	gsl_matrix_complex_add(params->H_q, params->env_term_56);

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

	}while(max_el > 1e-15);

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
	int i;

	gsl_matrix_complex_memcpy(params->rho, params->rho_init);

	for(i=0; i<params->time->size; i++)
	{	
		propagate(params->rho, gsl_vector_get(field, i), params);
		gsl_vector_set(params->pop_t_1, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 0, 0)));
		gsl_vector_set(params->pop_t_2, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 1, 1)));
		gsl_vector_set(params->pop_t_3, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 2, 2)));
		gsl_vector_set(params->pop_t_4, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 3, 3)));
		gsl_vector_set(params->pop_t_5, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 4, 4)));
		gsl_vector_set(params->pop_t_6, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 5, 5)));
		gsl_vector_set(params->pop_t_7, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 6, 6)));
		gsl_vector_set(params->pop_t_8, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 7, 7)));
		gsl_vector_set(params->pop_t_9, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 8, 8)));
		gsl_vector_set(params->pop_t_10, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 9, 9)));
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
		
		gsl_vector_set(params->pop_t_1, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 0, 0)));
		gsl_vector_set(params->pop_t_2, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 1, 1)));
		gsl_vector_set(params->pop_t_3, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 2, 2)));
		gsl_vector_set(params->pop_t_4, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 3, 3)));

		gsl_vector_set(params->pop_t_5, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 4, 4)));
		gsl_vector_set(params->pop_t_6, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 5, 5)));
		gsl_vector_set(params->pop_t_7, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 6, 6)));
		gsl_vector_set(params->pop_t_8, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 7, 7)));
		gsl_vector_set(params->pop_t_9, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 8, 8)));
		gsl_vector_set(params->pop_t_10, i, GSL_REAL(gsl_matrix_complex_get(params->rho, 9, 9)));
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
		gsl_vector_set(params->dJ_dE, tau_indx, GSL_REAL(gsl_matrix_complex_get(params->g, params->j_indx_max, params->j_indx_max)));
		// - GSL_REAL(gsl_matrix_complex_get(params->g, params->j_indx_min, params->j_indx_min)));
		propagate(params->rho, gsl_vector_get(field_func(A_phi_params, params), tau_indx), params);
	}

	for(j=0; j<Nfreq; j++)
	{
		double dA = 0.0;
		double dphi = 0.0;
		int k;
		for(tau_indx=0;  tau_indx<Tsteps;  tau_indx++)
		{
			double gaussian_t = exp(-(tau_indx*dt-params->time_central)*(tau_indx*dt-params->time_central)/(2.0*params->sigma2));
			//double sin_env = gsl_sf_pow_int(gsl_sf_sin(2.0*M_PI*tau_indx*dt/params->time_final),2);
			double x = gsl_sf_cos(gsl_vector_get(params->frequency, j)*tau_indx*dt + gsl_vector_get(A_phi_params, j+Nfreq));

			double y = -gsl_vector_get(A_phi_params, j)
				*gsl_sf_sin(gsl_vector_get(params->frequency, j)*tau_indx*dt + gsl_vector_get(A_phi_params, j+Nfreq));

			if(tau_indx == 0) k=1;
			else if(tau_indx == (Tsteps-1)) k=1;
			else if( tau_indx % 2 == 0) k=2;
			else if( tau_indx % 2 == 1) k=4;

			dA += gsl_vector_get(params->dJ_dE,tau_indx)*gaussian_t*x*k;
			dphi += gsl_vector_get(params->dJ_dE,tau_indx)*gaussian_t*y*k;
		
			//dA += gsl_vector_get(params->dJ_dE,tau_indx)*sin_env*x*k;
			//dphi += gsl_vector_get(params->dJ_dE,tau_indx)*sin_env*y*k;
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

int main()
{
	const int N = 10;
	int Nfreq = 4;
	int i;
	const int j_indx_max = 2;
	const int j_indx_min = 9;

	double freq_central = .4796679; 			// 625 nm in fs-1

	double gamma_41  = 1./(freq_central*150.);		// 150 fs
	double gamma_23  = 1./(freq_central*150.);		// 150 ps
	double gamma_35  = 1./(freq_central*61.*100.);		// 61. ps
	double gamma_34  = 1./(freq_central*26.*100.);		// 26. ps
	double gamma_56  = 1./(freq_central*100000.);		// 1 ms

	double gamma_96  = 1./(freq_central*10.);		// 10 fs
	double gamma_78  = 1./(freq_central*50.);    	 	// 50 fs
	double gamma_810 = 1./(freq_central*1.5*1000.);		// 1.5 ps
	double gamma_89  = 1./(freq_central*300.);		// 300 fs
	double gamma_101 = 1./(freq_central*100000.);		// 1 ms

	double gamma_dephasing = 1./(freq_central*300.);	// 30 fs

	const double dt = 1.;
	const int Tsteps = 250;
	const int Tsteps_tot = 1500;
	const int Tsteps_zero_field = Tsteps_tot-Tsteps;
	const double time_initial = 0.00;
	const double time_final = dt*Tsteps;
	const double time_central = (time_final-time_initial)/1.8;
	const double sigma2 = (time_final-time_initial)/0.20;

	gsl_vector *time = gsl_vector_alloc(Tsteps);
	for(i = 0; i<Tsteps; i++)
	{
		gsl_vector_set(time, i, time_initial+dt*i);
	}

	double detuning = 0.002;
	gsl_vector *A_phi_params = gsl_vector_alloc(2*Nfreq);

	const gsl_rng_type * T;
	gsl_rng * r;

	gsl_rng_env_setup();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);

	for(i=0; i<Nfreq; i++)
	{

		double u = gsl_rng_uniform (r);
		gsl_vector_set(A_phi_params, i, 0.1*u);
		gsl_vector_set(A_phi_params, i+Nfreq, 0.00);
	}


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

	print_vec(frequency);

	double ds = 0.5;
	int Ssteps = 8;
	double s_initial = 0.0;

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
	gsl_matrix_complex_set(rho_init, 0, 0, gsl_complex_rect(0.5, 0.0));
	gsl_matrix_complex_set(rho_init, 5, 5, gsl_complex_rect(0.5, 0.0));

	parameters params;
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
	params.j_indx_max = j_indx_max;
	params.j_indx_min = j_indx_min;
	params.count = 0;
	params.time_central = time_central;
	params.sigma2 = sigma2;
	params.dJ_dA_dphi = gsl_vector_alloc(2*Nfreq);
	params.frequency = frequency;

	params.gamma_41 = gamma_41;
	params.gamma_23 = gamma_23;
	params.gamma_35 = gamma_35;
	params.gamma_34 = gamma_34;
	params.gamma_56 = gamma_56;

	params.gamma_96 = gamma_96;
	params.gamma_78 = gamma_78;
	params.gamma_89 = gamma_89;
	params.gamma_810 = gamma_810;
	params.gamma_101 = gamma_101;

	params.gamma_dephasing = gamma_dephasing;

	params.pop_t_1 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_2 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_3 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_4 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_5 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_6 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_7 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_8 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_9 = gsl_vector_alloc(Tsteps_tot);
	params.pop_t_10 = gsl_vector_alloc(Tsteps_tot);

	params.env_term_41  = gsl_matrix_complex_alloc(N, N);
	params.env_term_23  = gsl_matrix_complex_alloc(N, N);
	params.env_term_35  = gsl_matrix_complex_alloc(N, N);
	params.env_term_34  = gsl_matrix_complex_alloc(N, N);
	params.env_term_56  = gsl_matrix_complex_alloc(N, N);

	params.env_term_96  = gsl_matrix_complex_alloc(N, N);
	params.env_term_78  = gsl_matrix_complex_alloc(N, N);
	params.env_term_810 = gsl_matrix_complex_alloc(N, N);
	params.env_term_89  = gsl_matrix_complex_alloc(N, N);
	params.env_term_101 = gsl_matrix_complex_alloc(N, N);
	
	params.dephase_term = gsl_matrix_complex_alloc(N, N);

	FILE *field_t = fopen ("field_ini.out", "wb");
	gsl_vector_fwrite (field_t, field_func(A_phi_params, &params));
	fclose (field_t);
	
	gsl_matrix_complex_memcpy(params.rho, params.rho_init);

	gsl_vector *pop_1 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_2 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_3 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_4 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_5 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_6 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_7 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_8 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_9 = gsl_vector_alloc(Ssteps+1);
	gsl_vector *pop_10 = gsl_vector_alloc(Ssteps+1);

	gsl_vector *s_vec = gsl_vector_alloc(Ssteps+1);

	gsl_odeiv2_system sys = {Evolution, NULL, 2*params.frequency->size, &params};
	gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rkf45, 1e-6, 0.0, 1e-4);

	rho_propagate_0_T(field_func(A_phi_params, &params), &params);
	rho_propagate_T_T_total(&params);
	FILE *output_file_di_1 = fopen ("pop_dyn_ini_1.out", "wb");
	gsl_vector_fwrite (output_file_di_1, params.pop_t_1);
	fclose (output_file_di_1);

	FILE *output_file_di_2 = fopen ("pop_dyn_ini_2.out", "wb");
	gsl_vector_fwrite (output_file_di_2, params.pop_t_2);
	fclose (output_file_di_2);

	FILE *output_file_di_3 = fopen ("pop_dyn_ini_3.out", "wb");
	gsl_vector_fwrite (output_file_di_3, params.pop_t_3);
	fclose (output_file_di_3);

	FILE *output_file_di_4 = fopen ("pop_dyn_ini_4.out", "wb");
	gsl_vector_fwrite (output_file_di_4, params.pop_t_4);
	fclose (output_file_di_4);

	FILE *output_file_di_5 = fopen ("pop_dyn_ini_5.out", "wb");
	gsl_vector_fwrite (output_file_di_5, params.pop_t_5);
	fclose (output_file_di_5);

	FILE *output_file_di_6 = fopen ("pop_dyn_ini_6.out", "wb");
	gsl_vector_fwrite (output_file_di_6, params.pop_t_6);
	fclose (output_file_di_6);

	FILE *output_file_di_7 = fopen ("pop_dyn_ini_7.out", "wb");
	gsl_vector_fwrite (output_file_di_7, params.pop_t_7);
	fclose (output_file_di_7);

	FILE *output_file_di_8 = fopen ("pop_dyn_ini_8.out", "wb");
	gsl_vector_fwrite (output_file_di_8, params.pop_t_8);
	fclose (output_file_di_8);

	FILE *output_file_di_9 = fopen ("pop_dyn_ini_9.out", "wb");
	gsl_vector_fwrite (output_file_di_9, params.pop_t_9);
	fclose (output_file_di_9);

	FILE *output_file_di_10 = fopen ("pop_dyn_ini_10.out", "wb");
	gsl_vector_fwrite (output_file_di_10, params.pop_t_10);
	fclose (output_file_di_10);

	print_diag(params.rho);
	print_vec(A_phi_params);

	gsl_vector_set(pop_1, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 0, 0)));
	gsl_vector_set(pop_2, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 1, 1)));
	gsl_vector_set(pop_3, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 2, 2)));
	gsl_vector_set(pop_4, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 3, 3)));
	gsl_vector_set(pop_5, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 4, 4)));
	gsl_vector_set(pop_6, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 5, 5)));
	gsl_vector_set(pop_7, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 6, 6)));
	gsl_vector_set(pop_8, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 7, 7)));
	gsl_vector_set(pop_9, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 8, 8)));
	gsl_vector_set(pop_10, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 9, 9)));

	gsl_vector_set(s_vec, 0, 0.0);
	
	for(i=1; i<Ssteps+1; i++)
	{
		params.count = 0;
		double smin = s_initial + i*ds;
		double smax = smin + ds;

		printf("Iteration number %d \n", i);
		const int status = gsl_odeiv2_driver_apply (driver, &smin, smax, A_phi_params->data);
		
		if (status != GSL_SUCCESS)
		{
			printf ("Error (%d) occurred while solving D-MORPH ODEs !\n", status); break;	
		}

		rho_propagate_0_T(field_func(A_phi_params, &params), &params);
		rho_propagate_T_T_total(&params);
		print_diag(params.rho);
		print_vec(A_phi_params);

		gsl_vector_set(pop_1, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 0, 0)));
		gsl_vector_set(pop_2, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 1, 1)));
		gsl_vector_set(pop_3, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 2, 2)));
		gsl_vector_set(pop_4, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 3, 3)));
		gsl_vector_set(pop_5, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 4, 4)));
		gsl_vector_set(pop_6, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 5, 5)));
		gsl_vector_set(pop_7, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 6, 6)));
		gsl_vector_set(pop_8, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 7, 7)));
		gsl_vector_set(pop_9, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 8, 8)));
		gsl_vector_set(pop_10, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 9, 9)));

		gsl_vector_set(s_vec, i, i*ds);

	}

	FILE *output_file_1 = fopen ("population_evolution_1.out", "wb");
	gsl_vector_fwrite (output_file_1, pop_1);
	fclose (output_file_1);

	FILE *output_file_2 = fopen ("population_evolution_2.out", "wb");
	gsl_vector_fwrite (output_file_2, pop_2);
	fclose (output_file_2);

	FILE *output_file_3 = fopen ("population_evolution_3.out", "wb");
	gsl_vector_fwrite (output_file_3, pop_3);
	fclose (output_file_3);

	FILE *output_file_4 = fopen ("population_evolution_4.out", "wb");
	gsl_vector_fwrite (output_file_4, pop_4);
	fclose (output_file_4);

	FILE *output_file_5 = fopen ("population_evolution_5.out", "wb");
	gsl_vector_fwrite (output_file_5, pop_5);
	fclose (output_file_5);

	FILE *output_file_6 = fopen ("population_evolution_6.out", "wb");
	gsl_vector_fwrite (output_file_6, pop_6);
	fclose (output_file_6);

	FILE *output_file_7 = fopen ("population_evolution_7.out", "wb");
	gsl_vector_fwrite (output_file_7, pop_7);
	fclose (output_file_7);

	FILE *output_file_8 = fopen ("population_evolution_8.out", "wb");
	gsl_vector_fwrite (output_file_8, pop_8);
	fclose (output_file_8);

	FILE *output_file_9 = fopen ("population_evolution_9.out", "wb");
	gsl_vector_fwrite (output_file_9, pop_9);
	fclose (output_file_9);

	FILE *output_file_10 = fopen ("population_evolution_10.out", "wb");
	gsl_vector_fwrite (output_file_10, pop_10);
	fclose (output_file_10);

	FILE *output_s = fopen ("s.out", "wb");
	gsl_vector_fwrite (output_s, s_vec);
	fclose (output_s);

	FILE *output_t = fopen ("t.out", "wb");
	gsl_vector_fwrite (output_t, params.time);
	fclose (output_t);

	FILE *field_f = fopen ("field_out.out", "wb");
	gsl_vector_fwrite (field_f, field_func(A_phi_params, &params));
	fclose (field_f);

	FILE *field_params = fopen ("field_out_params.out", "wb");
	gsl_vector_fwrite (field_params, A_phi_params);
	fclose (field_params);

	FILE *output_file_df_1 = fopen ("pop_dyn_fin_1.out", "wb");
	gsl_vector_fwrite (output_file_df_1, params.pop_t_1);
	fclose (output_file_df_1);

	FILE *output_file_df_2 = fopen ("pop_dyn_fin_2.out", "wb");
	gsl_vector_fwrite (output_file_df_2, params.pop_t_2);
	fclose (output_file_df_2);

	FILE *output_file_df_3 = fopen ("pop_dyn_fin_3.out", "wb");
	gsl_vector_fwrite (output_file_df_3, params.pop_t_3);
	fclose (output_file_df_3);

	FILE *output_file_df_4 = fopen ("pop_dyn_fin_4.out", "wb");
	gsl_vector_fwrite (output_file_df_4, params.pop_t_4);
	fclose (output_file_df_4);

	FILE *output_file_df_5 = fopen ("pop_dyn_fin_5.out", "wb");
	gsl_vector_fwrite (output_file_df_5, params.pop_t_5);
	fclose (output_file_df_5);

	FILE *output_file_df_6 = fopen ("pop_dyn_fin_6.out", "wb");
	gsl_vector_fwrite (output_file_df_6, params.pop_t_6);
	fclose (output_file_df_6);

	FILE *output_file_df_7 = fopen ("pop_dyn_fin_7.out", "wb");
	gsl_vector_fwrite (output_file_df_7, params.pop_t_7);
	fclose (output_file_df_7);

	FILE *output_file_df_8 = fopen ("pop_dyn_fin_8.out", "wb");
	gsl_vector_fwrite (output_file_df_8, params.pop_t_8);
	fclose (output_file_df_8);
	gsl_rng_free (r);

	FILE *output_file_df_9 = fopen ("pop_dyn_fin_9.out", "wb");
	gsl_vector_fwrite (output_file_df_9, params.pop_t_9);
	fclose (output_file_df_9);

	FILE *output_file_df_10 = fopen ("pop_dyn_fin_10.out", "wb");
	gsl_vector_fwrite (output_file_df_10, params.pop_t_10);
	fclose (output_file_df_10);

	return GSL_SUCCESS;
}

