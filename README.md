GraMat (Graphic Card Matrix Interface) is a library that I (Dr. Yuan-Yen Tai) developed during my Ph.D study. This library can help boost the use of the MAGMA library which solve linear algebra equations. GraMat integrate single, double, single-complex and double-complex floating variable and provide an easy and unified interface for the basic matrix operation.
Up to now, the most important operations are:
1. Matrix multiplication,
2. General matrix eigenvalue decomposition,
3. Symmetric/Hermitian matrix eigenvalue decomposition.

GraMat relies on the MAGMA GPU library, it also takes the LAPACK wrap for the CPU calculation, and it is very simple to switch in between GPU and CPU matrix operations with GraMat.

The source code of GraMat is written in header files and there is no need to pre-compile the library itself. However, in order to use GraMat, one have to install the following libraries:

1. The CUDA driver and SDK.
2. The MAGMA library.

After the successful compilation of MAGMA library, one can use GraMat by including the library path in the compile flag: 
`-I/[path to gramat folder]/include`.
