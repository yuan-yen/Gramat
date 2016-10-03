#include <iostream>
#include <vector>
#include <unistd.h>
using namespace std;

// includes, project

#include "gramat.h"

// Single matrix diagonalization example
void S_EigTest() {
	gmt::NGPU = 2;
	gmt::smat S(2,2),V;
	gmt::smat E;

	S(0,0) = 1;
	S(1,0) = 2;
	S(0,1) = 2;
	S(1,1) = 1;

	cout<<S.evd(E,V)<<endl;

	cout<<S<<endl;
	cout<<E<<endl;
	cout<<V<<endl;
}

// Double matrix diagonalization example
void D_EigTest() {
	gmt::dmat S(2,2),V;
	gmt::dmat E;
	S(0,0) = 1;
	S(1,0) = 2+1;
	S(0,1) = 2-1;
	S(1,1) = 1;

	cout<<S.evd(E,V)<<endl;

	cout<<S<<endl;
	cout<<E<<endl;
	cout<<V<<endl;
}

// Single complex matrix diagonalization example
void C_EigTest() {
	gmt::cmat S(2,2),V;
	gmt::smat E;
	S(0,0) = 1;
	S(1,0) = 2+gmt::Im;
	S(0,1) = 2-gmt::Im;
	S(1,1) = 1;

	cout<<S.evd(E,V)<<endl;

	cout<<S<<endl;
	cout<<E<<endl;
	cout<<V<<endl;
}

// Double complex matrix diagonalization example
void Z_EigTest() {
	gmt::zmat S(2,2),V;
	gmt::dmat E;
	S(0,0) = 1;
	S(1,0) = 2+gmt::Im;
	S(0,1) = 2-gmt::Im;
	S(1,1) = 1;

	cout<<S.evd(E,V)<<endl;

	//cout<<S<<endl;
	cout<<E<<endl;
	cout<<V<<endl;
}

// Single matrix multiplication example
void magmaSmul() {
	gmt::smat A(2,3), B(3,2), C;

	A(0,0)=1;
	A(0,1)=2;
	A(0,2)=0;
	A(1,0)=1;
	A(1,1)=3;
	A(1,2)=1;

	B(0,0)=1;
	B(0,1)=2;
	B(1,0)=0;
	B(1,1)=1;
	B(2,0)=3;
	B(2,1)=1;


	cout<<multiply(A,B,C)<<endl;
	cout<<A<<endl;
	cout<<B<<endl;
	cout<<C<<endl;
}

// Double matrix multiplication example
void magmaDmul() {
	gmt::dmat A(2,3), B(3,2), C;

	A(0,0)=1;
	A(0,1)=2;
	A(0,2)=0;
	A(1,0)=1;
	A(1,1)=3;
	A(1,2)=1;

	B(0,0)=1;
	B(0,1)=2;
	B(1,0)=0;
	B(1,1)=1;
	B(2,0)=3;
	B(2,1)=1;


	cout<<multiply(A,B,C)<<endl;

	cout<<A<<endl;
	cout<<B<<endl;
	cout<<C<<endl;
}

// Single complex matrix multiplication example
void magmaCmul() {
	gmt::cmat A(2,3), B(3,2), C;

	A(0,0)=1;
	A(0,1)=2;
	A(0,2)=0;
	A(1,0)=1;
	A(1,1)=3;
	A(1,2)=1;

	B(0,0)=1;
	B(0,1)=2;
	B(1,0)=0;
	B(1,1)=1;
	B(2,0)=3;
	B(2,1)=1;


	cout<<multiply(A,B,C)<<endl;

	cout<<A<<endl;
	cout<<B<<endl;
	cout<<C<<endl;
}

// Double complex matrix multiplication example
void magmaZmul() {
	gmt::zmat A(2,3), B(3,2), C;

	A(0,0)=1;
	A(0,1)=2;
	A(0,2)=0;
	A(1,0)=1;
	A(1,1)=3;
	A(1,2)=1;

	B(0,0)=1;
	B(0,1)=2;
	B(1,0)=0;
	B(1,1)=1;
	B(2,0)=3;
	B(2,1)=1;


	cout<<multiply(A,B,C)<<endl;

	cout<<A<<endl;
	cout<<B<<endl;
	cout<<C<<endl;
}

#define max(a,b) (((a)<(b))?(b):(a))

int main(int argc, char **argv){
	
	gmt::PRINT_PRECISION = 10;

	gmt::LINEAR_SOLVER = gmt::LAPACK;
	//gmt::LINEAR_SOLVER = gmt::MAGMA;
	
	//S_EigTest();
	//D_EigTest();
	//C_EigTest();
	//Z_EigTest();
	//magmaSmul();
	//magmaDmul();
	//magmaCmul();
	magmaZmul();
	
	//gmt::dmat A(2,2), V;
	//gmt::zmat E;
	//
	//A(0,0) = 1;
	//A(0,1) = 2;
	//A(1,0) = 2;
	//A(1,1) = 1;
	//
	//A.gevd(E,V);
	//cout<<E<<endl;
	//cout<<V<<endl;
	
	
	return 0;
}

