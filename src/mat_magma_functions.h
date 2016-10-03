
/*********************************************************************/

double	magma_smul(unsigned M, unsigned K, unsigned N, float *a, float *b, float *c)							{
	// Calling magma_sgemm

	magma_int_t m = M;
	magma_int_t k = K;
	magma_int_t n = N;
	

	float alpha = MAGMA_S_MAKE(1.0, 0.0);
	float beta	= MAGMA_S_MAKE(0.0, 0.0);

	real_Double_t calc_time = magma_wtime();
	
	switch(SOLVER){
		case CPU:
			blasf77_sgemm("N", "N", &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
			break;
		case GPU: case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			// device mem. for a, b, c
			magma_init();
			magma_int_t mk = m*k;
			magma_int_t kn = k*n;
			magma_int_t mn = m*n;
			float	*d_a, *d_b, *d_c;
			magma_smalloc( &d_a, mk );
			magma_smalloc( &d_b, kn );
			magma_smalloc( &d_c, mn );

			// copy data from host to device
			magma_ssetmatrix( m, k, a, m, d_a, m );
			magma_ssetmatrix( k, n, b, k, d_b, k );
			magma_ssetmatrix( m, n, c, m, d_c, m );

			magma_sgemm(MagmaNoTrans,MagmaNoTrans,m,n,k,alpha,d_a,m,d_b,k, beta,d_c,m);

			magma_sgetmatrix( m, n, d_c, m, c, m );

			magma_free(d_a);
			magma_free(d_b);
			magma_free(d_c);
			magma_finalize ();
			break;
	}
	
	calc_time = magma_wtime()-calc_time;
	return calc_time;
}
double	magma_dmul(unsigned M, unsigned K, unsigned N, double *a, double *b, double *c)							{
	// Calling magma_dgemm

	magma_int_t m = M;
	magma_int_t k = K;
	magma_int_t n = N;
	double	alpha	= MAGMA_D_MAKE(1.0, 0.0);
	double	beta	= MAGMA_D_MAKE(0.0, 0.0);

	real_Double_t calc_time = magma_wtime();
	
	switch(SOLVER){
		case CPU:
			blasf77_dgemm("N", "N", &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
			break;
		case GPU: case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			// device mem. for a, b, c
			magma_init();
			magma_int_t mk = m*k;
			magma_int_t kn = k*n;
			magma_int_t mn = m*n;
			double	*d_a, *d_b, *d_c;
			magma_dmalloc( &d_a, mk );
			magma_dmalloc( &d_b, kn );
			magma_dmalloc( &d_c, mn );

			// copy data from host to device
			magma_dsetmatrix( m, k, a, m, d_a, m );
			magma_dsetmatrix( k, n, b, k, d_b, k );
			magma_dsetmatrix( m, n, c, m, d_c, m );

			magma_dgemm(MagmaNoTrans,MagmaNoTrans,m,n,k,alpha,d_a,m,d_b,k, beta,d_c,m);

			magma_dgetmatrix( m, n, d_c, m, c, m );

			magma_free(d_a);
			magma_free(d_b);
			magma_free(d_c);
			magma_finalize ();
			break;
	}
	
	calc_time = magma_wtime()-calc_time;
	return calc_time;
}
double	magma_cmul(unsigned M, unsigned K, unsigned N, cuFloatComplex *a, cuFloatComplex *b, cuFloatComplex *c)	{
	// Calling magma_cgemm

	magma_int_t m = M;
	magma_int_t k = K;
	magma_int_t n = N;

	cuFloatComplex 	alpha	= MAGMA_C_MAKE(1.0, 0.0);
	cuFloatComplex	beta	= MAGMA_C_MAKE(0.0, 0.0);

	real_Double_t calc_time = magma_wtime();
	
	switch(SOLVER){
		case CPU:
			blasf77_cgemm("N", "N", &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
			break;
		case GPU: case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			// device mem. for a, b, c
			magma_init();
			magma_int_t mk = m*k;
			magma_int_t kn = k*n;
			magma_int_t mn = m*n;
			cuFloatComplex	*d_a, *d_b, *d_c;
			magma_cmalloc( &d_a, mk );
			magma_cmalloc( &d_b, kn );
			magma_cmalloc( &d_c, mn );

			// copy data from host to device
			magma_csetmatrix( m, k, a, m, d_a, m );
			magma_csetmatrix( k, n, b, k, d_b, k );
			magma_csetmatrix( m, n, c, m, d_c, m );

			magma_cgemm(MagmaNoTrans,MagmaNoTrans,m,n,k,alpha,d_a,m,d_b,k, beta,d_c,m);

			magma_cgetmatrix( m, n, d_c, m, c, m );

			magma_free(d_a);
			magma_free(d_b);
			magma_free(d_c);
			magma_finalize ();
			break;
	}

	calc_time = calc_time-magma_wtime();
	return calc_time;
}
double	magma_zmul(unsigned M, unsigned K, unsigned N, cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c){
	// Calling magma_zgemm

	magma_int_t m = M;
	magma_int_t k = K;
	magma_int_t n = N;

	cuDoubleComplex 	alpha=MAGMA_Z_MAKE(1.0, 0.0);
	cuDoubleComplex		beta= MAGMA_Z_MAKE(0.0, 0.0);

	real_Double_t calc_time = magma_wtime();
	
	switch(SOLVER){
		case CPU:
			blasf77_zgemm("N", "N", &m, &n, &k, &alpha, a, &m, b, &k, &beta, c, &m);
			break;
		case GPU: case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			// device mem. for a, b, c
			magma_init();
			magma_int_t mk = m*k;
			magma_int_t kn = k*n;
			magma_int_t mn = m*n;
			cuDoubleComplex	*d_a, *d_b, *d_c;
			magma_zmalloc( &d_a, mk );
			magma_zmalloc( &d_b, kn );
			magma_zmalloc( &d_c, mn );

			// copy data from host to device
			magma_zsetmatrix( m, k, a, m, d_a, m );
			magma_zsetmatrix( k, n, b, k, d_b, k );
			magma_zsetmatrix( m, n, c, m, d_c, m );

			magma_zgemm(MagmaNoTrans,MagmaNoTrans,m,n,k,alpha,d_a,m,d_b,k, beta,d_c,m);

			magma_zgetmatrix( m, n, d_c, m, c, m );

			magma_free(d_a);
			magma_free(d_b);
			magma_free(d_c);
			magma_finalize ();
			break;
	}

	calc_time = calc_time-magma_wtime();
	return calc_time;
}

/*********************************************************************/

double	magma_evd(unsigned N, float *A, float *w1, float *h_A, string &Status)								{
	//-----------------------------------
	//magma_ssyevd
	//------------------------------------

	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();

	magma_int_t n=N;
	float  *h_work;	// single precision
	magma_int_t lwork = -1, *iwork, liwork = -1, info;


	lapackf77_slacpy( MagmaUpperLowerStr, &n, &n, A, &n, h_A, &n ); // single precision, copy A->h_A
	
	
	float  aux_work [1]; // single precision
	magma_int_t aux_iwork[1];
	
	switch(SOLVER){
		case CPU:
			// Query for workspace sizes

			lapackf77_ssyevd("V","L"            ,&n,h_A,&n,w1,aux_work ,&lwork, aux_iwork ,&liwork,&info ); //Single prcision query for dimension

			lwork = (magma_int_t) aux_work[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_smalloc_cpu(&h_work,lwork); //memory query
			
			lapackf77_ssyevd("V", "L"           , &n, h_A, &n, w1, h_work,&lwork, iwork,&liwork,&info); // Perform eigen-value problem
			break;
		case GPU:
			magma_init ();
			// Query for workspace sizes
			magma_ssyevd    (MagmaVec,MagmaLower, n,h_A,n,w1,aux_work , -1   , aux_iwork ,-1    ,&info ); //Single prcision query for dimension

			lwork = (magma_int_t) aux_work[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_smalloc_cpu(&h_work,lwork); //memory query

			// Perform eigen-value problem
			magma_ssyevd    (MagmaVec,MagmaLower,  n, h_A,  n, w1, h_work, lwork, iwork, liwork,&info);

			magma_finalize();
			break;
		case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			magma_init ();
			// Query for workspace sizes
			magma_ssyevd_m(SOLVER,MagmaVec,MagmaLower,n,h_A,n,w1,aux_work ,-1, aux_iwork ,-1,&info ); //Single prcision query for dimension

			lwork = (magma_int_t) aux_work[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_smalloc_cpu(&h_work,lwork); //memory query

			// Perform eigen-value problem
			magma_ssyevd_m(SOLVER,MagmaVec,MagmaLower,n,h_A,n,w1,h_work,lwork, iwork,liwork,&info);

			magma_finalize();
			break;
	}
	
	free(h_work);

	if (info != 0)  Status=magma_strerror( info );
	else			Status="Success";

	// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	return calc_time;
}
double	magma_evd(unsigned N, double *A, double *w1, double *h_A, string &Status)							{
	//-----------------------------------
	//magma_dsyevd
	//-----------------------------------

	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();

	magma_int_t n=N;
	double *h_work;	// double precision
	magma_int_t lwork = -1, *iwork, liwork = -1, info;


	lapackf77_dlacpy( MagmaUpperLowerStr, &n, &n, A, &n, h_A, &n ); // double precision, copy A->h_A

	// Query for workspace sizes
	double aux_work [1]; // double precision
	magma_int_t aux_iwork[1];

	
	switch(SOLVER){
		case CPU:
			lapackf77_dsyevd("V","L"            ,&n,h_A,&n,w1,aux_work ,&lwork, aux_iwork ,&liwork,&info ); //Single prcision query for dimension

			lwork = (magma_int_t) aux_work[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_dmalloc_cpu(&h_work,lwork); //memory query

			lapackf77_dsyevd("V", "L"           , &n, h_A, &n, w1, h_work,&lwork, iwork,&liwork,&info); // Perform eigen-value problem
			break;
		case GPU:
			magma_init ();
			magma_dsyevd(MagmaVec,MagmaLower,n,h_A,n,w1,aux_work ,-1, aux_iwork ,-1,&info ); //Double prcision query for dimension

			lwork = (magma_int_t) aux_work[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_dmalloc_cpu(&h_work,lwork); //memory query

			// Perform eigen-value problem
			magma_dsyevd(MagmaVec,MagmaLower,n,h_A,n,w1,h_work,lwork, iwork,liwork,&info);

			magma_finalize();
		case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			magma_init ();
			magma_dsyevd_m(SOLVER, MagmaVec,MagmaLower,n,h_A,n,w1,aux_work ,-1, aux_iwork ,-1,&info ); //Double prcision query for dimension

			lwork = (magma_int_t) aux_work[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_dmalloc_cpu(&h_work,lwork); //memory query

			// Perform eigen-value problem
			magma_dsyevd_m(SOLVER,MagmaVec,MagmaLower,n,h_A,n,w1,h_work,lwork, iwork,liwork,&info);

			magma_finalize();
			break;
	}
	
	free(h_work);
	
	if (info != 0)  Status=magma_strerror( info );
	else			Status="Success";

	// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	return calc_time;
}
double	magma_evd(unsigned N, cuFloatComplex *A, float *w1, cuFloatComplex *h_A, string &Status)			{
	//-----------------------------------
	//magma_cheevd
	//-----------------------------------

	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();

	magma_int_t n=N;
	cuFloatComplex *h_work;	// single complex precision
	magma_int_t lwork = -1, *iwork, liwork = -1, lrwork = -1, info;
	float aux_rwork[1],*rwork;

	lapackf77_clacpy( MagmaUpperLowerStr, &n, &n, A, &n, h_A, &n ); // double precision, copy A->h_A

	// Query for workspace sizes
	cuFloatComplex aux_work [1]; // double precision
	magma_int_t aux_iwork[1];

	
	switch(SOLVER){
		case CPU:
			lapackf77_cheevd("N", "L", &n, h_A, &n, w1, aux_work, &lwork, aux_rwork, &lrwork, aux_iwork, &liwork, &info);

			lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
			lrwork = (magma_int_t) aux_rwork[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_cmalloc_cpu(&h_work,lwork); //memory query
			magma_smalloc_cpu(&rwork,lrwork); //memory query

			lapackf77_cheevd( "N", "L", &n, h_A, &n, w1, h_work, &lwork, rwork, &lrwork, iwork, &liwork, &info ); // Perform eigen-value problem

			break;
		case GPU:
			magma_init ();
			magma_cheevd(MagmaVec, MagmaLower, n,h_A,n,w1	,aux_work ,-1 ,aux_rwork,-1 ,aux_iwork,-1 ,&info ); //single complex prcision query for dimension
			
			lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
			lrwork = (magma_int_t) aux_rwork[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_cmalloc_cpu(&h_work,lwork); //memory query
			magma_smalloc_cpu(&rwork,lrwork); //memory query
			
			magma_cheevd( MagmaVec, MagmaLower, n, h_A, n, w1, h_work, lwork, rwork, lrwork, iwork, liwork, &info ); // Perform eigen-value problem
			
			magma_finalize();
			break;
		case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			magma_init ();
			magma_cheevd_m(SOLVER,MagmaVec, MagmaLower, n,h_A,n,w1	,aux_work ,-1 ,aux_rwork,-1 ,aux_iwork,-1 ,&info ); //single complex prcision query for dimension
			
			lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
			lrwork = (magma_int_t) aux_rwork[0];
			liwork = aux_iwork[0];
			
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_cmalloc_cpu(&h_work,lwork); //memory query
			magma_smalloc_cpu(&rwork,lrwork); //memory query
			
			// Perform eigen-value problem
			magma_cheevd_m(SOLVER,MagmaVec, MagmaLower, n, h_A, n, w1, h_work, lwork, rwork, lrwork, iwork, liwork, &info );
			
			magma_finalize();
			break;
	}
	

	free(h_work);
	free(rwork);
	
	if (info != 0)  Status=magma_strerror( info );
	else			Status="Success";

	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	return calc_time;
}
double	magma_evd(unsigned N, cuDoubleComplex *A, double *w1, cuDoubleComplex *h_A, string &Status)			{
	//-----------------------------------
	//magma_zheevd
	//-----------------------------------

	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();

	magma_int_t n=N;
	cuDoubleComplex *h_work;	// single complex precision
	magma_int_t lwork = -1, *iwork, liwork = -1, lrwork = -1, info;
	double	aux_rwork[1],*rwork;

	lapackf77_zlacpy( MagmaUpperLowerStr, &n, &n, A, &n, h_A, &n ); // double precision, copy A->h_A

	// Query for workspace sizes
	cuDoubleComplex aux_work [1]; // double precision
	magma_int_t aux_iwork[1];
	
	switch(SOLVER){
		case CPU:
			lapackf77_zheevd("N", "L", &n, h_A, &n, w1, aux_work, &lwork, aux_rwork, &lrwork, aux_iwork, &liwork, &info);
			
			lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
			lrwork = (magma_int_t) aux_rwork[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_zmalloc_cpu(&h_work,lwork); //memory query
			magma_dmalloc_cpu(&rwork,lrwork); //memory query
			
			// Perform eigen-value problem
			lapackf77_zheevd( "N", "L", &n, h_A, &n, w1, h_work, &lwork, rwork, &lrwork, iwork, &liwork, &info );
			
			magma_finalize();
			break;
		case GPU: case GPUx2: case GPUx3: case GPUx4: case GPUx5:
			magma_init ();
			magma_zheevd(MagmaVec, MagmaLower, n,h_A,n,w1	,aux_work ,-1 ,aux_rwork,-1 ,aux_iwork,-1 ,&info ); //single complex prcision query for dimension
			
			lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
			lrwork = (magma_int_t) aux_rwork[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_zmalloc_cpu(&h_work,lwork); //memory query
			magma_dmalloc_cpu(&rwork,lrwork); //memory query
			
			magma_zheevd( MagmaVec, MagmaLower, n, h_A, n, w1, h_work, lwork, rwork, lrwork, iwork, liwork, &info ); // Perform eigen-value problem
			
			magma_finalize();
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			magma_init ();
			magma_zheevd_m(1,MagmaVec, MagmaLower, n,h_A,n,w1	,aux_work ,-1 ,aux_rwork,-1 ,aux_iwork,-1 ,&info ); //single complex prcision query for dimension
			
			lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
			lrwork = (magma_int_t) aux_rwork[0];
			liwork = aux_iwork[0];
			iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
			magma_zmalloc_cpu(&h_work,lwork); //memory query
			magma_dmalloc_cpu(&rwork,lrwork); //memory query
			
			magma_zheevd_m(1,MagmaVec, MagmaLower, n, h_A, n, w1, h_work, lwork, rwork, lrwork, iwork, liwork, &info ); // Perform eigen-value problem
			
			magma_finalize();
			break;
	}


	free(h_work);
	free(rwork);

	if (info != 0)  Status=magma_strerror( info );
	else			Status="Success";

	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	return calc_time;
}

/*********************************************************************/

double	magma_gevd(unsigned N, float *A, float *wr, float *wi, float *VR, string &Status)					{
	//-----------------------------------
	//magma_sgeev
	//-----------------------------------
	
	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();
	
	magma_init ();
	magma_int_t n=N, n2=n*n;
	float *VL;
	
	magma_int_t info, nb;
	float *h_work;
	magma_int_t lwork;
	nb = magma_get_sgehrd_nb(n);
	lwork = n*(2+2*nb);
	lwork = max(lwork, n*(5+2*n));
	
	magma_smalloc_cpu(&VL,n2);
	magma_smalloc_cpu(&h_work, lwork);
	
	switch(SOLVER){
		case CPU:
			lapackf77_sgeev("N", "V", &n, A, &n, wr, wi, VL, &n, VR, &n, h_work, &lwork, &info);
			break;
		case GPU: case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			magma_sgeev(MagmaNoVec, MagmaVec, n, A, n, wr, wi, VL, n, VR, n, h_work, lwork, &info);
			//magma_sgeev_m(MagmaNoVec, MagmaVec, n, A, n, wr, wi, VL, n, VR, n, h_work, lwork, &info);
			break;
	}
	
	free(VL);
	free(h_work);
	magma_finalize();
	
	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	
	return calc_time;
}
double	magma_gevd(unsigned N, double *A, double *wr, double *wi, double *VR, string &Status)				{
	//-----------------------------------
	//magma_dgeev
	//-----------------------------------
	
	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();
	
	magma_init ();
	magma_int_t n=N, n2=n*n;
	double *VL;
	
	magma_int_t info, nb;
	double *h_work;
	magma_int_t lwork;
	nb = magma_get_sgehrd_nb(n);
	lwork = n*(2+2*nb);
	lwork = max(lwork, n*(5+2*n));
	
	magma_dmalloc_cpu(&VL,n2);
	magma_dmalloc_cpu(&h_work, lwork);
	
	switch(SOLVER){
		case CPU:
			lapackf77_dgeev("N", "V", &n, A, &n, wr, wi, VL, &n, VR, &n, h_work, &lwork, &info);
			break;
		case GPU: case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			magma_dgeev(MagmaNoVec, MagmaVec, n, A, n, wr, wi, VL, n, VR, n, h_work, lwork, &info);
			//magma_dgeev_m(MagmaNoVec, MagmaVec, n, A, n, wr, wi, VL, n, VR, n, h_work, lwork, &info);
			break;
	}
	
	
	free(VL);
	free(h_work);
	magma_finalize();
	
	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	
	return calc_time;
}
double	magma_gevd(unsigned N, cuFloatComplex *A, cuFloatComplex *w, cuFloatComplex *VR, string &Status)	{
	//-----------------------------------
	//magma_cgeev
	//-----------------------------------
	
	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();
	
	magma_init ();
	magma_int_t n=N, n2=n*n;
	cuFloatComplex *VL;
	
	magma_int_t info, nb;
	cuFloatComplex *h_work;
	magma_int_t lwork;
	nb = magma_get_sgehrd_nb(n);
	lwork = n*(1+2*nb);
	lwork = max(lwork, n*(5+2*n));
	float *rwork;
	
	magma_cmalloc_cpu(&VL,n2);
	magma_cmalloc_cpu(&h_work, lwork);
	magma_smalloc_cpu(&rwork,n2);
	
	switch(SOLVER){
		case CPU:
			lapackf77_cgeev("N", "V", &n, A, &n, w, VL, &n, VR, &n, h_work, &lwork, rwork, &info);
			break;
		case GPU: case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			magma_cgeev(MagmaNoVec, MagmaVec,n , A, n, w, VL, n, VR, n, h_work, lwork, rwork, &info);
			//magma_cgeev_m(MagmaNoVec, MagmaVec,n , A, n, w, VL, n, VR, n, h_work, lwork, rwork, &info);
			break;
	}
	
	
	
	free(VL);
	free(h_work);
	free(rwork);
	magma_finalize();
	
	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	
	return calc_time;
}
double	magma_gevd(unsigned N, cuDoubleComplex *A, cuDoubleComplex *w, cuDoubleComplex *VR, string &Status)	{
	//-----------------------------------
	//magma_zgeev
	//-----------------------------------
	
	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();
	
	magma_init ();
	magma_int_t n=N, n2=n*n;
	cuDoubleComplex *VL;
	
	magma_int_t info, nb;
	cuDoubleComplex *h_work;
	magma_int_t lwork;
	nb = magma_get_sgehrd_nb(n);
	lwork = n*(1+2*nb);
	lwork = max(lwork, n*(5+2*n));
	double *rwork;
	
	magma_zmalloc_cpu(&VL,n2);
	magma_zmalloc_cpu(&h_work, lwork);
	magma_dmalloc_cpu(&rwork,n2);
	
	switch(SOLVER){
		case CPU:
			lapackf77_zgeev("N", "V", &n, A, &n, w, VL, &n, VR, &n, h_work, &lwork, rwork, &info);
			break;
		case GPU:
		case GPUx2: case GPUx3: case GPUx4: case GPUx5:
		case GPUx6: case GPUx7: case GPUx8: case GPUx9: case GPUx10:
			magma_zgeev(MagmaNoVec, MagmaVec,n , A, n, w, VL, n, VR, n, h_work, lwork, rwork, &info);
			//magma_zgeev_m(MagmaNoVec, MagmaVec,n , A, n, w, VL, n, VR, n, h_work, lwork, rwork, &info);
			break;
	}
	
	
	free(VL);
	free(h_work);
	free(rwork);
	magma_finalize();
	
	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	
	return calc_time;
}

