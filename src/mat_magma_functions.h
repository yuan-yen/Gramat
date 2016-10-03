
// Single GPU operations
//-----------------------------------
//magma_ssyevd
//------------------------------------
float magma_evd(unsigned int N, float *A, float *w1, float *h_A, string &Status) {

	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();


	magma_init ();
	magma_int_t n=N;
	float  *h_work;	// single precision
	magma_int_t lwork = -1, *iwork, liwork = -1, info;


	lapackf77_slacpy( MagmaUpperLowerStr, &n, &n, A, &n, h_A, &n ); // single precision, copy A->h_A

	// Query for workspace sizes
	float  aux_work [1]; // single precision
	magma_int_t aux_iwork[1];

	magma_ssyevd    (MagmaVec,MagmaLower, n,h_A,n,w1,aux_work , -1   , aux_iwork ,-1    ,&info ); //Single prcision query for dimension
	//lapackf77_ssyevd("V","L"            ,&n,h_A,&n,w1,aux_work ,&lwork, aux_iwork ,&liwork,&info ); //Single prcision query for dimension
	//magma_ssyevd_m(1,MagmaVec,MagmaLower,n,h_A,n,w1,aux_work ,-1, aux_iwork ,-1,&info ); //Single prcision query for dimension

	lwork = (magma_int_t) aux_work[0];
	liwork = aux_iwork[0];

	iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	magma_smalloc_cpu(&h_work,lwork); //memory query

	// Perform eigen-value problem
	magma_ssyevd    (MagmaVec,MagmaLower,  n, h_A,  n, w1, h_work, lwork, iwork, liwork,&info);
	//lapackf77_ssyevd("V", "L"           , &n, h_A, &n, w1, h_work,&lwork, iwork,&liwork,&info);
	//magma_ssyevd_m(1,MagmaVec,MagmaLower,n,h_A,n,w1,h_work,lwork, iwork,liwork,&info);

	free(h_work);
	magma_finalize();


	if (info != 0)  Status=magma_strerror( info );
	else			Status="Success";

	// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	return calc_time;
}

//-----------------------------------
//magma_dsyevd
//-----------------------------------
float magma_evd(unsigned int N, double *A, double *w1, double *h_A, string &Status) {

	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();

	magma_init ();
	magma_int_t n=N;
	double *h_work;	// double precision
	magma_int_t lwork = -1, *iwork, liwork = -1, info;


	lapackf77_dlacpy( MagmaUpperLowerStr, &n, &n, A, &n, h_A, &n ); // double precision, copy A->h_A

	// Query for workspace sizes
	double aux_work [1]; // double precision
	magma_int_t aux_iwork[1];

	magma_dsyevd(MagmaVec,MagmaLower,n,h_A,n,w1,aux_work ,-1, aux_iwork ,-1,&info ); //Double prcision query for dimension
	//lapackf77_dsyevd("V","L"            ,&n,h_A,&n,w1,aux_work ,&lwork, aux_iwork ,&liwork,&info ); //Single prcision query for dimension
	//magma_dsyevd_m(1, MagmaVec,MagmaLower,n,h_A,n,w1,aux_work ,-1, aux_iwork ,-1,&info ); //Double prcision query for dimension

	lwork = (magma_int_t) aux_work[0];
	liwork = aux_iwork[0];

	iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	magma_dmalloc_cpu(&h_work,lwork); //memory query

	// Perform eigen-value problem
	magma_dsyevd(MagmaVec,MagmaLower,n,h_A,n,w1,h_work,lwork, iwork,liwork,&info);
	//lapackf77_dsyevd("V", "L"           , &n, h_A, &n, w1, h_work,&lwork, iwork,&liwork,&info);
	//magma_dsyevd_m(1,MagmaVec,MagmaLower,n,h_A,n,w1,h_work,lwork, iwork,liwork,&info);

	free(h_work);
	magma_finalize();


	if (info != 0)  Status=magma_strerror( info );
	else			Status="Success";

	// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	return calc_time;
}

//-----------------------------------
//magma_cheevd
//-----------------------------------
float magma_evd(unsigned int N, cuFloatComplex *A, float *w1, cuFloatComplex *h_A, string &Status) {

	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();

	magma_init ();
	magma_int_t n=N;
	cuFloatComplex *h_work;	// single complex precision
	magma_int_t lwork = -1, *iwork, liwork = -1, lrwork = -1, info;
	float aux_rwork[1],*rwork;

	lapackf77_clacpy( MagmaUpperLowerStr, &n, &n, A, &n, h_A, &n ); // double precision, copy A->h_A

	// Query for workspace sizes
	cuFloatComplex aux_work [1]; // double precision
	magma_int_t aux_iwork[1];

	magma_cheevd(MagmaVec, MagmaLower, n,h_A,n,w1	,aux_work ,-1 ,aux_rwork,-1 ,aux_iwork,-1 ,&info ); //single complex prcision query for dimension
	//lapackf77_cheevd("N", "L", &n, h_A, &n, w1, aux_work, &lwork, aux_rwork, &lrwork, aux_iwork, &liwork, &info);
	//magma_cheevd_m(1,MagmaVec, MagmaLower, n,h_A,n,w1	,aux_work ,-1 ,aux_rwork,-1 ,aux_iwork,-1 ,&info ); //single complex prcision query for dimension

    lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
    lrwork = (magma_int_t) aux_rwork[0];
    liwork = aux_iwork[0];

	iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	magma_cmalloc_cpu(&h_work,lwork); //memory query
	magma_smalloc_cpu(&rwork,lrwork); //memory query

	// Perform eigen-value problem
	magma_cheevd( MagmaVec, MagmaLower, n, h_A, n, w1, h_work, lwork, rwork, lrwork, iwork, liwork, &info );
	//lapackf77_cheevd( "N", "L", &n, h_A, &n, w1, h_work, &lwork, rwork, &lrwork, iwork, &liwork, &info );
	//magma_cheevd_m(1,MagmaVec, MagmaLower, n, h_A, n, w1, h_work, lwork, rwork, lrwork, iwork, liwork, &info );

	free(h_work);
	free(rwork);
	magma_finalize();

	if (info != 0)  Status=magma_strerror( info );
	else			Status="Success";

	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	return calc_time;
}

//-----------------------------------
//magma_zheevd
//-----------------------------------
float magma_evd(unsigned int N, cuDoubleComplex *A, double *w1, cuDoubleComplex *h_A, string &Status) {

	// Start counting the calculation time
	real_Double_t calc_time = magma_wtime();

	magma_init ();
	magma_int_t n=N;
	cuDoubleComplex *h_work;	// single complex precision
	magma_int_t lwork = -1, *iwork, liwork = -1, lrwork = -1, info;
	double	aux_rwork[1],*rwork;

	lapackf77_zlacpy( MagmaUpperLowerStr, &n, &n, A, &n, h_A, &n ); // double precision, copy A->h_A

	// Query for workspace sizes
	cuDoubleComplex aux_work [1]; // double precision
	magma_int_t aux_iwork[1];

	magma_zheevd(MagmaVec, MagmaLower, n,h_A,n,w1	,aux_work ,-1 ,aux_rwork,-1 ,aux_iwork,-1 ,&info ); //single complex prcision query for dimension
	//lapackf77_zheevd("N", "L", &n, h_A, &n, w1, aux_work, &lwork, aux_rwork, &lrwork, aux_iwork, &liwork, &info);
	//magma_zheevd_m(1,MagmaVec, MagmaLower, n,h_A,n,w1	,aux_work ,-1 ,aux_rwork,-1 ,aux_iwork,-1 ,&info ); //single complex prcision query for dimension

	lwork  = (magma_int_t) MAGMA_C_REAL( aux_work[0] );
	lrwork = (magma_int_t) aux_rwork[0];
	liwork = aux_iwork[0];

	iwork=(magma_int_t*)malloc(liwork*sizeof(magma_int_t));
	magma_zmalloc_cpu(&h_work,lwork); //memory query
	magma_dmalloc_cpu(&rwork,lrwork); //memory query

	// Perform eigen-value problem
	magma_zheevd( MagmaVec, MagmaLower, n, h_A, n, w1, h_work, lwork, rwork, lrwork, iwork, liwork, &info );
	//lapackf77_zheevd( "N", "L", &n, h_A, &n, w1, h_work, &lwork, rwork, &lrwork, iwork, &liwork, &info );
	//magma_zheevd_m(1,MagmaVec, MagmaLower, n, h_A, n, w1, h_work, lwork, rwork, lrwork, iwork, liwork, &info );

	free(h_work);
	free(rwork);
	magma_finalize();

	if (info != 0)  Status=magma_strerror( info );
	else			Status="Success";

	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	return calc_time;
}


/*********************************************************************/


//-----------------------------------
//magma_sgeev
//-----------------------------------
float magma_gevd(unsigned int N, float *A, float *wr, float *wi, float *VR, string &Status)	{
	
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
	
	magma_sgeev(MagmaNoVec, MagmaVec, n, A, n, wr, wi, VL, n, VR, n, h_work, lwork, &info);
	//lapackf77_sgeev("N", "V", &n, A, &n, wr, wi, VL, &n, VR, &n, h_work, &lwork, &info);
	//magma_sgeev_m(MagmaNoVec, MagmaVec, n, A, n, wr, wi, VL, n, VR, n, h_work, lwork, &info);
	
	free(VL);
	free(h_work);
	magma_finalize();
	
	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	
	return calc_time;
}

//-----------------------------------
//magma_dgeev
//-----------------------------------
float magma_gevd(unsigned int N, double *A, double *wr, double *wi, double *VR, string &Status)	{
	
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
	
	magma_dgeev(MagmaNoVec, MagmaVec, n, A, n, wr, wi, VL, n, VR, n, h_work, lwork, &info);
	//lapackf77_dgeev("N", "V", &n, A, &n, wr, wi, VL, &n, VR, &n, h_work, &lwork, &info);
	//magma_dgeev_m(MagmaNoVec, MagmaVec, n, A, n, wr, wi, VL, n, VR, n, h_work, lwork, &info);
	
	free(VL);
	free(h_work);
	magma_finalize();
	
	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	
	return calc_time;
}

//-----------------------------------
//magma_cgeev
//-----------------------------------
float magma_gevd(unsigned int N, cuFloatComplex *A, cuFloatComplex *w, cuFloatComplex *VR, string &Status)	{
	
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
	
	magma_cgeev(MagmaNoVec, MagmaVec,n , A, n, w, VL, n, VR, n, h_work, lwork, rwork, &info);
	//lapackf77_cgeev("N", "V", &n, A, &n, w, VL, &n, VR, &n, h_work, &lwork, rwork, &info);
	//magma_cgeev_m(MagmaNoVec, MagmaVec,n , A, n, w, VL, n, VR, n, h_work, lwork, rwork, &info);
	
	
	
	free(VL);
	free(h_work);
	free(rwork);
	magma_finalize();
	
	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	
	return calc_time;
}

//-----------------------------------
//magma_zgeev
//-----------------------------------
float magma_gevd(unsigned int N, cuDoubleComplex *A, cuDoubleComplex *w, cuDoubleComplex *VR, string &Status)	{
	
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
	
	magma_zgeev(MagmaNoVec, MagmaVec,n , A, n, w, VL, n, VR, n, h_work, lwork, rwork, &info);
	//lapackf77_zgeev("N", "V", &n, A, &n, w, VL, &n, VR, &n, h_work, &lwork, rwork, &info);
	//magma_zgeev_m(MagmaNoVec, MagmaVec,n , A, n, w, VL, n, VR, n, h_work, lwork, rwork, &info);
	
	free(VL);
	free(h_work);
	free(rwork);
	magma_finalize();
	
	//// End counting the calculation time
	calc_time = magma_wtime() - calc_time;
	
	return calc_time;
}









