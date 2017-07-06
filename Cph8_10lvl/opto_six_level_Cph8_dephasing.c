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
	double time_central;
	double sigma2;
	gsl_vector *frequency;

	gsl_vector *dJ_dE;
	const gsl_vector *A_phi_params;
	gsl_vector *dJ_dA_dphi;

	int j_indx;
	int count;

	double gamma_65;
	double gamma_54;
	double gamma_41;
	double gamma_43;
	double gamma_34;
	double gamma_32;
	double gamma_dephasing;

	gsl_matrix_complex *env_term_65;
	gsl_matrix_complex *env_term_54;
	gsl_matrix_complex *env_term_41;
	gsl_matrix_complex *env_term_43;
	gsl_matrix_complex *env_term_34;
	gsl_matrix_complex *env_term_32;
	gsl_matrix_complex *dephase_term;

	gsl_vector *pop_t_1;
	gsl_vector *pop_t_2;
	gsl_vector *pop_t_3;
	gsl_vector *pop_t_4;
	gsl_vector *pop_t_5;
	gsl_vector *pop_t_6;


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
	const double gamma_65 = params->gamma_65;
	const double gamma_54 = params->gamma_41;
	const double gamma_41 = params->gamma_65;
	const double gamma_43 = params->gamma_43;
	const double gamma_34 = params->gamma_34;
	const double gamma_32 = params->gamma_32;

	environment_term(qmat, 4, 5, gamma_65, params->env_term_65, params); 
	environment_term(qmat, 3, 4, gamma_54, params->env_term_54, params);
	environment_term(qmat, 0, 3, gamma_41, params->env_term_41, params); 
	environment_term(qmat, 2, 3, gamma_43, params->env_term_43, params);
	environment_term(qmat, 3, 2, gamma_34, params->env_term_34, params); 
	environment_term(qmat, 1, 2, gamma_32, params->env_term_32, params);
	dephasing_term(qmat, params->dephase_term, params);

	const gsl_complex set_one = gsl_complex_rect(1.0, 0.0);
	const gsl_complex set_zero = gsl_complex_rect(0.0, 0.0);

	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, params->Htot, qmat, set_zero, params->H_q);
	gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, set_one, qmat, params->Htot, set_zero, params->q_H);

	gsl_matrix_complex_sub(params->H_q, params->q_H);
	gsl_matrix_complex_scale(params->H_q, gsl_complex_rect(0.0, -1.0));

	gsl_matrix_complex_add(params->H_q, params->env_term_65);
	gsl_matrix_complex_add(params->H_q, params->env_term_54);
	gsl_matrix_complex_add(params->H_q, params->env_term_41);
	gsl_matrix_complex_add(params->H_q, params->env_term_43);
	gsl_matrix_complex_add(params->H_q, params->env_term_34);
	gsl_matrix_complex_add(params->H_q, params->env_term_32);
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
	const int N = 6;
	int Nfreq = 10;
	int i;
	const int j_indx = 1;

	double freq_central = 3.0138; 				// in fs-1

	double gamma_65 = .225;
	double gamma_54 = .6756;
	double gamma_41 = .1126;
	double gamma_43 = .1126;
	double gamma_34 = .519;
	double gamma_32 = .1126;

	double gamma_dephasing = 1./(freq_central*30.);		// 30 fs

	const double dt = 1.0;
	const int Tsteps = 300;
	const double time_initial = 0.00;
	const double time_final = dt*Tsteps;
	const double time_central = (time_final-time_initial)/1.8;
	const double sigma2 = (time_final-time_initial)/0.20;

	gsl_vector *time = gsl_vector_alloc(Tsteps);
	for(i = 0; i<Tsteps; i++)
	{
		gsl_vector_set(time, i, time_initial+dt*i);
	}

	double detuning_p = 0.002;
	double detuning_s = 0.002;
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


	double E[6];
	E[0] = 0.00;
	E[1] = 0.05;
	E[2] = 0.85;
	E[3] = 0.90;
	E[4] = 0.95;
	E[5] = 1.00;

	gsl_vector *frequency = gsl_vector_alloc(Nfreq);
	gsl_vector_set(frequency, 0, E[5]-E[0]+2.0*detuning_p);
	gsl_vector_set(frequency, 1, E[5]-E[0]+detuning_p);
	gsl_vector_set(frequency, 2, E[5]-E[0]);
	gsl_vector_set(frequency, 3, E[5]-E[0]-detuning_p);
	gsl_vector_set(frequency, 4, E[5]-E[0]-2.0*detuning_p);

	gsl_vector_set(frequency, 5, E[2]-E[1]+2.0*detuning_s);
	gsl_vector_set(frequency, 6, E[2]-E[1]+detuning_s);
	gsl_vector_set(frequency, 7, E[2]-E[1]);
	gsl_vector_set(frequency, 8, E[2]-E[1]-detuning_s);
	gsl_vector_set(frequency, 9, E[2]-E[1]-2.0*detuning_s);

	print_vec(frequency);

	double ds = 0.5;
	int Ssteps = 30;
	double s_initial = 0.0;

	gsl_matrix_complex *H0 = gsl_matrix_complex_calloc(N, N);
	gsl_matrix_complex_set(H0, 0, 0, gsl_complex_rect(E[0], 0.0));
	gsl_matrix_complex_set(H0, 1, 1, gsl_complex_rect(E[1], 0.0));
	gsl_matrix_complex_set(H0, 2, 2, gsl_complex_rect(E[2], 0.0));
	gsl_matrix_complex_set(H0, 3, 3, gsl_complex_rect(E[3], 0.0));
	gsl_matrix_complex_set(H0, 4, 4, gsl_complex_rect(E[4], 0.0));
	gsl_matrix_complex_set(H0, 5, 5, gsl_complex_rect(E[5], 0.0));

	gsl_matrix_complex *mu_init = gsl_matrix_complex_calloc(N, N);
	gsl_matrix_complex_set(mu_init, 0, 5, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 5, 0, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 1, 2, gsl_complex_rect(1.0, 0.0));
	gsl_matrix_complex_set(mu_init, 2, 1, gsl_complex_rect(1.0, 0.0));

	gsl_matrix_complex *rho_init = gsl_matrix_complex_calloc(N, N);
	gsl_matrix_complex_set(rho_init, 0, 0, gsl_complex_rect(1.0, 0.0));

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
	params.A_phi_params = gsl_vector_alloc(2*Nfreq);
	params.field_plus = gsl_vector_alloc(Tsteps);
	params.field_minus = gsl_vector_alloc(Tsteps);
	params.dJ_dE = gsl_vector_alloc(Tsteps);
	params.j_indx = j_indx;
	params.count = 0;
	params.time_central = time_central;
	params.sigma2 = sigma2;
	params.dJ_dA_dphi = gsl_vector_alloc(2*Nfreq);
	params.frequency = frequency;

	params.gamma_65 = gamma_65;
	params.gamma_54 = gamma_54;
	params.gamma_41 = gamma_41;
	params.gamma_43 = gamma_43;
	params.gamma_34 = gamma_34;
	params.gamma_32 = gamma_32;
	params.gamma_dephasing = gamma_dephasing;

	params.pop_t_1 = gsl_vector_alloc(Tsteps);
	params.pop_t_2 = gsl_vector_alloc(Tsteps);
	params.pop_t_3 = gsl_vector_alloc(Tsteps);
	params.pop_t_4 = gsl_vector_alloc(Tsteps);
	params.pop_t_5 = gsl_vector_alloc(Tsteps);
	params.pop_t_6 = gsl_vector_alloc(Tsteps);

	params.env_term_65 = gsl_matrix_complex_alloc(N, N);
	params.env_term_54 = gsl_matrix_complex_alloc(N, N);
	params.env_term_41 = gsl_matrix_complex_alloc(N, N);
	params.env_term_43 = gsl_matrix_complex_alloc(N, N);
	params.env_term_34 = gsl_matrix_complex_alloc(N, N);
	params.env_term_32 = gsl_matrix_complex_alloc(N, N);
	
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
	gsl_vector *s_vec = gsl_vector_alloc(Ssteps+1);

	gsl_odeiv2_system sys = {Evolution, NULL, 2*params.frequency->size, &params};
	gsl_odeiv2_driver *driver = gsl_odeiv2_driver_alloc_y_new (&sys, gsl_odeiv2_step_rkf45, 1e-6, 0.0, 1e-4);

	rho_propagate_0_T(field_func(A_phi_params, &params), &params);

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

	print_cmplx_mat(params.rho);
	print_vec(A_phi_params);

	gsl_vector_set(pop_1, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 0, 0)));
	gsl_vector_set(pop_2, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 1, 1)));
	gsl_vector_set(pop_3, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 2, 2)));
	gsl_vector_set(pop_4, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 3, 3)));
	gsl_vector_set(pop_5, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 4, 4)));
	gsl_vector_set(pop_6, 0, GSL_REAL(gsl_matrix_complex_get(params.rho, 5, 5)));

	gsl_vector_set(s_vec, 0, 0.0);
	
	for(i=1; i<Ssteps+1; i++)
	{
		params.count = 0;
		double smin = s_initial + i*ds;
		double smax = smin + ds;

		printf("Iteration number %d \n", i+1);
		const int status = gsl_odeiv2_driver_apply (driver, &smin, smax, A_phi_params->data);
		
		if (status != GSL_SUCCESS)
		{
			printf ("Error (%d) occurred while solving D-MORPH ODEs !\n", status); break;	
		}

		rho_propagate_0_T(field_func(A_phi_params, &params), &params);
		print_cmplx_mat(params.rho);
		print_vec(A_phi_params);

		gsl_vector_set(pop_1, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 0, 0)));
		gsl_vector_set(pop_2, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 1, 1)));
		gsl_vector_set(pop_3, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 2, 2)));
		gsl_vector_set(pop_4, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 3, 3)));
		gsl_vector_set(pop_5, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 4, 4)));
		gsl_vector_set(pop_6, i, GSL_REAL(gsl_matrix_complex_get(params.rho, 5, 5)));

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

	FILE *output_s = fopen ("s.out", "wb");
	gsl_vector_fwrite (output_s, s_vec);
	fclose (output_s);

	FILE *output_t = fopen ("t.out", "wb");
	gsl_vector_fwrite (output_t, params.time);
	fclose (output_t);

	FILE *field_f = fopen ("field_out.out", "wb");
	gsl_vector_fwrite (field_f, field_func(A_phi_params, &params));
	fclose (field_f);

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

	gsl_rng_free (r);
	return GSL_SUCCESS;
}

