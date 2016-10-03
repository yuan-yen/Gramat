#ifndef __GRAMAT_HEADER__
#define __GRAMAT_HEADER__

//// includes, system
#include <stdio.h>
#include <stdlib.h>

// STD library
#include <string>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <complex>
#include <cmath>
#include <memory>

// includes, magma / cuda library
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cublas_v2.h>
#include "magma.h"
#include "magma_lapack.h"
//#include "magma_clapack.h"

namespace gmt {
	
	using namespace std;
		
	void err(string s){
		cout<<s<<endl;
		throw s;
	}
	
	unsigned PRINT_PRECISION=8;
	unsigned NGPU = 1;
	
	#include "mat_complex.h"
	#include "mat_fmt.h"
	
	/* ---------------------------------------------
	--- Class for data storage and transfer. -------
	-----------------------------------------------*/
	template<class T> class Arr {
		T * arr;
	public:
		unsigned _cols,_rows;
		Arr	(unsigned N=0): arr(new T[N])		{	}
		~Arr	()								{ delete [] arr;}
		T	&	at(unsigned i)					{return arr[i];}
		T *		get_ptr()						{return arr;}
	};
	
	/* ---------------------------------------------
	--- Class for operations of matrix. --------------
	-----------------------------------------------*/
	template<class var> class matrix;
	
	typedef matrix<svar>	smat;
	typedef matrix<dvar>	dmat;
	typedef matrix<cvar>	cmat;
	typedef matrix<zvar>	zmat;
	
	//--------------------------------------------------------------
	//---Some math functions
	//--------------------------------------------------------------
	double					abs(cvar val)		{ return sqrt(val.real()*val.real()+val.imag()*val.imag()); }
	double					abs(zvar val)		{ return sqrt(val.real()*val.real()+val.imag()*val.imag()); }
	zvar					sqrt(zvar _X)		{
		complex<double> X(_X.x,_X.y);
		X=sqrt(X);
		return zvar(X.real(),X.imag());
	}
	cvar					sqrt(cvar _X)		{
		complex<float> X(_X.x,_X.y);
		X=sqrt(X);
		return cvar(X.real(),X.imag());
	}
	zvar					exp(zvar val)		{
		zmplx vv(val.x,val.y);
		vv=exp(vv);
		return zvar(vv.real(), vv.imag());
	}
	
	enum type{MATRIX, ARRAY};
	
	enum solver{
		CPU,
		GPU,
		GPUx2,
		GPUx3,
		GPUx4,
		GPUx5,
		GPUx6,
		GPUx7,
		GPUx8,
		GPUx9,
		GPUx10,
	} SOLVER = GPU;
	
	#include "mat_magma_functions.h"
	#include "mat.h"
	
}

#endif
