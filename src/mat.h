/*-----------------------------------------------------------|
| Copyright (C) 2016 Yuan-Yen Tai                            |
|                                                            |
| This file is distributed under the terms of the GNU        |
| GENERAL PUBLIC LICENSE:                                    |
|    https://www.gnu.org/licenses/gpl-3.0.en.html            |
|                                                            |
|-----------------------------------------------------------*/
//
//  GraMat
//
//  Created by Yuan-Yen Tai
//

template<class var>
class matrix {
private:
	type			_TYPE;
	unsigned int	print_len;
	unsigned int	print_flag;
	void			set_col		(unsigned  c)	{mat->_cols=c;}
	void			set_row		(unsigned  r)	{mat->_rows=r;}

	auto_ptr<Arr<var> >		mat;
public:
	matrix(unsigned C=1, type tp=MATRIX): mat(new Arr<var>(C*C))			{
		mat->_cols=C; mat->_rows=C; zerolize();
		print_len = 10;
		print_flag=0;
		_TYPE=tp;
	}
	matrix(unsigned C, unsigned R, type tp=MATRIX): mat(new Arr<var>(C*R))	{
		mat->_cols=C; mat->_rows=R; zerolize();
		print_len = 12;
		print_flag=0;
		_TYPE=tp;
	}
	matrix(const matrix &  cm):mat(new Arr<var>(cm.cols()*cm.rows()))		{
		mat->_cols = cm.cols();
		mat->_rows = cm.rows();
		_TYPE = cm._TYPE;
		print_len = cm.print_len;
		print_flag= cm.print_flag;
		for( unsigned i=0 ; i<size() ; i++) index(i)=cm.const_index(i);
	}

	/* ---------------------------------------------
	--- Setting the Type of the data base. ---------
	-----------------------------------------------*/
	type				TYPE	()					{return _TYPE;}
	matrix &			toA		()					{_TYPE=ARRAY;	return *this;}
	matrix &			toM		()					{_TYPE=MATRIX;	return *this;}
	void				set_type(type T)			{_TYPE = T;}
	void				zerolize ()					{ for (unsigned int i=0 ; i<cols()*rows() ; i++) index(i)=0; }

	/* ---------------------------------------------
	--- Copy operators -----------------------------
	-----------------------------------------------*/
	matrix &			operator=(matrix rhs) {
		
		{	//// Copy the entire data base to another instance.
	
	        auto_ptr<Arr<var> > _mat(new Arr<var>(rhs.cols()*rhs.rows()));
	        _mat->_cols = rhs.cols();
	        _mat->_rows = rhs.rows();
	
	        for (unsigned i=0 ; i < rhs.size(); i++){ _mat->at(i) = rhs.const_index(i); }
	        mat = _mat;
	
		}
		return *this;
	}
	matrix &			operator=(auto_ptr<Arr<var> > rhs){
		set_col(rhs->_cols);
		set_row(rhs->_rows);
		mat = rhs;
		return *this;
	}
	

	/* ---------------------------------------------
	--- rows/cols/size access ----------------------
	-----------------------------------------------*/
	unsigned			cols		()		const	{return mat->_cols;}
	unsigned			rows		()		const	{return mat->_rows;}
	unsigned			size		()		const	{return cols()*rows();}


	/* ---------------------------------------------
	--- Pointer operation for data transfer. -------
	-----------------------------------------------*/
	var *				get_ptr		(){return mat->get_ptr();}
	auto_ptr<Arr<var> > get			(){return mat;}

	/* ---------------------------------------------
	--- Indexing operation -------------------------
	-----------------------------------------------*/
	var &				operator[]	(unsigned i)		{
		if (i>=cols()*rows())	{
			string ErrorStr = "Error, index out of range of matrix element";
			throw ErrorStr;
		}
	return mat->at(i);
	}
    var &				index		(unsigned i)		{
		if (i>=cols()*rows())			{
			string ErrorStr = "Error, index out of range of matrix element";
			throw ErrorStr;
		}
	return mat->at(i);
	}
    var &				const_index (unsigned i) const	{
		if (i>=cols()*rows())	{
			string ErrorStr = "Error, index out of range of matrix element";
			throw ErrorStr;
		}
		return mat->at(i);
	}
	var &				operator()	(unsigned i, unsigned j){
		return at(i,j);
	}
	
	///* ------------------------------------------------------- */
	///* -Index accessing of the operator ---------------------- */
	///* ------------------------------------------------------- */
	var &				at(unsigned int i, unsigned int j){
	
		/* Indexing operation for the 2D matrix */
		if (i>=cols()) {throw "Index i out of range.";}
		if (j>=rows()) {throw "Index j out of range.";}
		return index(j*cols()+i);
	}
	matrix&				operator-()	{
		matrix  *ret = new matrix(cols(),rows());

		for (int i=0 ; i< size() ; i++) ret->index(i) = -index(i);

		return *ret;
	}

	void				setPrintFlag(unsigned flag){print_flag=flag;}
	void				setPrintLength(unsigned len){print_len = len;}
	unsigned			PrintLength(){return print_len;}
	unsigned			PrintFlag(){return print_flag;}

	///* --------------------------------------------- */
	///* - The entry of the gevd --------------------- */
	///* - Which diagonalize symmetric matrix--------- */
	///* --------------------------------------------- */
	string				evd (smat & E, matrix & V) {
		if (cols() != rows()) throw "Object is not a square matrix!";
		E = smat(1,cols()).get();
		V = matrix(cols(),cols()).get();
	
		string Status;
		magma_evd(cols(), mat->get_ptr(), E.get_ptr(), V.get_ptr(), Status);
		return Status;
	}
	string				evd (dmat & E, matrix & V) {
		if (cols() != rows()) throw "Object is not a square matrix!";
		E = dmat(1,cols()).get();
		V = matrix(cols(),cols()).get();
	
		string Status;
		magma_evd(cols(), mat->get_ptr(), E.get_ptr(), V.get_ptr(), Status);
		return Status;
	}
	string				gevd(cmat & E, smat & VL) {
		if (cols() != rows()) throw "Object is not a square matrix!";
		E	= cmat(1,cols()).get();
		smat A = *this;
		smat WR(1,cols());
		smat WI(1,cols());
		VL	= matrix(cols(),cols());
	
		string Status;
		magma_gevd(cols(), A.get_ptr(), WR.get_ptr(), WI.get_ptr(), VL.get_ptr(), Status);
		
		for(unsigned i=0 ; i<E.size() ; i++){
			E[i] = WR[i] + Im*WI[i];
		}
		
		return Status;
	}
	string				gevd(zmat & E, dmat & VL) {
		if (cols() != rows()) throw "Object is not a square matrix!";
		E	= zmat(1,cols()).get();
		dmat A = *this;
		dmat WR(1,cols());
		dmat WI(1,cols());
		VL	= dmat(cols(),cols());
	
		string Status;
		magma_gevd(cols(), A.get_ptr(), WR.get_ptr(), WI.get_ptr(), VL.get_ptr(), Status);
		
		for(unsigned i=0 ; i<E.size() ; i++){
			E[i] = WR[i] + Im*WI[i];
		}
		
		return Status;
	}
	string				gevd(cmat & E, cmat & VL) {
		if (cols() != rows()) throw "Object is not a square matrix!";
		E	= cmat(1,cols()).get();
		cmat A = *this;
		VL	= cmat(cols(),cols());
	
		string Status;
		magma_gevd(cols(), A.get_ptr(), E.get_ptr(), VL.get_ptr(), Status);
		
		return Status;
	}
	string				gevd(zmat & E, zmat & VL) {
		if (cols() != rows()) throw "Object is not a square matrix!";
		E	= zmat(1,cols()).get();
		zmat A = *this;
		VL	= zmat(cols(),cols());
	
		string Status;
		magma_gevd(cols(), A.get_ptr(), E.get_ptr(), VL.get_ptr(), Status);
		
		return Status;
	}

	matrix<var> &		col			(unsigned int index)	{
		///* ------------------------------------------------------- */
		///* -Column access----------------------------------------- */
		///* ------------------------------------------------------- */
		if (index>=cols()) {throw "Index out of range.";}
	
		matrix<var> * ret = new matrix<var>(cols(),1);
	
		for( unsigned int i=0 ; i<cols(); i++) ret->at(i,0) = at(i,index);
	
		return *ret;
	}
	matrix<var> &		row			(unsigned int index)	{
		///* ------------------------------------------------------- */
		///* -Row access-------------------------------------------- */
		///* ------------------------------------------------------- */
		if (index>=rows()) {throw "Index out of range.";}
	
		matrix<var> * ret = new matrix<var>(1,rows());
	
		for( unsigned int i=0 ; i<rows(); i++) ret->at(0,i) = at(index,i);
	
		return *ret;
	}
	void				col_swap	(unsigned  i,unsigned j){
		///* ------------------------------------------------------- */
		///* -Swap two columns-------------------------------------- */
		///* ------------------------------------------------------- */
		if (i >=rows()) {throw "Index out of range.";}
		if (j >=rows()) {throw "Index out of range.";}
	
		for( unsigned index=0 ; index<cols(); index++){
			var buffer=at(i,index);
			at(i,index) = at(j,index);
			at(j,index) = buffer;
		}
	}
	void				row_swap	(unsigned  i,unsigned j){
		///* ------------------------------------------------------- */
		///* -Swap two rows----------------------------------------- */
		///* ------------------------------------------------------- */
		if (i >=cols()) {throw "Index out of range.";}
		if (j >=cols()) {throw "Index out of range.";}
	
		for( unsigned index=0 ; index<rows(); index++){
			var buffer=at(index,i);
			at(index,i) = at(index,j);
			at(index,j) = buffer;
		}
	}
	
	matrix operator+(matrix &	rhs){
		if (cols() != rhs.cols()) err( "Adding in different _cols");
		if (rows() != rhs.rows()) err( "Adding in different _rows");

		matrix<var>  ret(cols(),rows());

		for (int i=0 ; i< cols()*rows() ; i++) ret.index(i) = index(i)+rhs[i];

		return ret;
	}
	matrix operator-(matrix &	rhs){
		if (cols() != rhs.cols()) err( "Adding in different _cols");
		if (rows() != rhs.rows()) err( "Adding in different _rows");

		matrix<var>  ret(cols(),rows());

		for (int i=0 ; i< cols()*rows() ; i++) ret.index(i) = index(i)-rhs[i];

		return ret;
	}
};

double							multiply(smat & A, smat & B, smat & C){
	if( A.size() ==0 or B.size() ==0 )	return -1;
	if( A.rows() != B.cols() )			return -1;

	C = smat(A.cols(), B.rows()).get();

	return magma_smul(A.cols(), A.rows(), B.rows(), A.get_ptr(), B.get_ptr(), C.get_ptr());
}
double							multiply(dmat & A, dmat & B, dmat & C){
	if( A.size() ==0 or B.size() ==0 )	return -1;
	if( A.rows() != B.cols() )			return -1;

	C = dmat(A.cols(), B.rows()).get();

	return magma_dmul(A.cols(), A.rows(), B.rows(), A.get_ptr(), B.get_ptr(), C.get_ptr());
}
double							multiply(cmat & A, cmat & B, cmat & C){
	if( A.size() ==0 or B.size() ==0 )	return -1;
	if( A.rows() != B.cols() )			return -1;

	C = cmat(A.cols(), B.rows()).get();

	return magma_cmul(A.cols(), A.rows(), B.rows(), A.get_ptr(), B.get_ptr(), C.get_ptr());
}
double							multiply(zmat & A, zmat & B, zmat & C){
	if( A.size() ==0 or B.size() ==0 )	return -1;
	if( A.rows() != B.cols() )			return -1;

	C = zmat(A.cols(), B.rows()).get();

	return magma_zmul(A.cols(), A.rows(), B.rows(), A.get_ptr(), B.get_ptr(), C.get_ptr());
}

/* -------------------------------------------------------
-- Adding operation --------------------------------------
---------------------------------------------------------*/
template<class var, class T>	matrix<var> operator+(matrix<var> & lhs, T rhs)				{
	matrix<var>		ret( lhs.cols(),lhs.rows() );
	for (int i=0 ; i< lhs.cols()*lhs.rows() ; i++) ret.index(i) = lhs[i]+rhs;
	return ret;
}
template<class var, class T>	matrix<var> operator+(T lhs,matrix<var> & rhs)				{
	matrix<var>		ret( rhs.cols(),rhs.rows() );
	for (int i=0 ; i< rhs.cols()*rhs.rows() ; i++) ret.index(i) = rhs[i]+lhs;
	return ret;
}

/* ------------------------------------------------------
-- Substract operation --------------------------------------
---------------------------------------------------------*/
template<class var, class T>	matrix<var> operator-(matrix<var> & lhs, T rhs)				{
	matrix<var>		ret( lhs.cols(),lhs.rows() );
	for (int i=0 ; i< lhs.cols()*lhs.rows() ; i++) ret.index(i) = lhs[i]-rhs;
	return ret;
}
template<class var, class T>	matrix<var> operator-(T lhs,matrix<var> & rhs)				{
	matrix<var>		ret( rhs.cols(),rhs.rows() );
	for (int i=0 ; i< rhs.cols()*rhs.rows() ; i++) ret.index(i) = lhs-rhs[i];
	return ret;
}

/* -------------------------------------------------------
-- Multiply operation ------------------------------------
---------------------------------------------------------*/
template<class var>				matrix<var> operator*(matrix<var> & lhs, matrix<var> & rhs)	{

	matrix<var>  ret(lhs.cols(),rhs.rows() );

	if		(lhs.TYPE() == ARRAY	and rhs.TYPE() == ARRAY){
		if (lhs.cols() != rhs.cols() and lhs.rows() != rhs.cols()) err( "Array dimension does not match!");
		for (unsigned int i=0 ; i< lhs.cols()*lhs.rows() ; i++){ ret.index(i) = lhs[i]*rhs[i]; }
	}
	else if	(lhs.TYPE() == MATRIX	and rhs.TYPE() == MATRIX){
		if (lhs.rows() != rhs.cols()) err( "Matrix does not match in k-index!");
		multiply(lhs, rhs, ret); // Call Magma_multipy function
	}
	else if	(lhs.TYPE() == ARRAY	and rhs.TYPE() == MATRIX)	err( "Data type does not match: lhs->Array & rhs->Matrix");
	else if	(lhs.TYPE() == MATRIX	and rhs.TYPE() == ARRAY)	err( "Data type does not match: lhs->Matrix & rhs->ARRAY");
	
	ret.set_type(lhs.TYPE());

	return ret;
}
template<class var, class T>	matrix<var> operator*(matrix<var> & lhs, T rhs)				{
	matrix<var>		ret( lhs.cols(),lhs.rows() );
	for (int i=0 ; i< lhs.cols()*lhs.rows() ; i++) ret.index(i) = lhs[i]*rhs;
	return ret;
}
template<class var, class T>	matrix<var> operator*(T lhs,matrix<var> & rhs)				{
	matrix<var>		ret( rhs.cols(),rhs.rows() );
	for (int i=0 ; i< rhs.cols()*rhs.rows() ; i++) ret.index(i) = lhs*rhs[i];
	return ret;
}

//--- Vector operations ---
template<class vv> matrix<vv>	curl(matrix<vv> a, matrix<vv> b)		{
	matrix<vv> z(1,3);
	if (a.size()==3 and b.size()==3 ) {
		z[0]=a[1]*b[2]-a[2]*b[1];
		z[1]=a[2]*b[0]-a[0]*b[2];
		z[2]=a[0]*b[1]-a[1]*b[0];
	}
	return z;
}
template<class vv> vv			cdot(matrix<vv> a, matrix<vv> b)		{
	vv z=0;
	if (a.size()==3 and b.size()==3 ) {
		z+=a[0]*b[0];
		z+=a[1]*b[1];
		z+=a[2]*b[2];
	}
	return z;
}

///* ------------------------------------------------------- */
///* -Output of the matrix---------------------------------- */
///* ------------------------------------------------------- */
template<class var>				ostream & operator<< (ostream & out, matrix<var> & m)		{
	
	// Determine the max length of each component
	int max_len=m.PrintLength();
	for (int i=0 ; i< m.cols() ; i++)
	for (int j=0 ; j< m.rows() ; j++){
		string ss = tostr(m(i,j));
		if (max_len < ss.length() )
			max_len = ss.length();
	}
	

	for (unsigned int i=0 ; i< m.cols() ; i++){
		if ( i>0 )out<<endl;
		for (unsigned int j=0 ; j< m.rows() ; j++){
			if (i==0 and j==0) out<<"[[ ";
			else if (j==0) out<<" [ ";
			else out<<" ";
			
			out<<fformat(m(i,j), max_len+1);
			
			if (i==m.cols()-1 and j==m.rows()-1) out<<" ]]";
			else if (j==m.rows()-1) out<<" ] ";
		}
	}
	return out;
}

//--------------------------------------------------------------
//---Perform Gram-Schmidt orthogonalization to the matrix-------
//--------------------------------------------------------------
template<class vvar> void		Orthonormalize(matrix<vvar> & a, matrix<vvar> & b)			{
	b=matrix<vvar>(1,a.cols()).get();

	for(int i=0 ; i<a.cols() ; i++)
	{
		if(i>0)
		for( int j=0 ; j<i ; j++)
		{
			vvar c1=0;
			for( int k=0 ; k<a.cols() ; k++) { c1=c1+conj(a(k,j))*a(k,i); }
			for( int k=0 ; k<a.cols() ; k++) { a(k,i)=a(k,i)-c1*a(k,j);	  }
		}

		b[i]=0;
		for( int k=0 ; k<a.cols() ; k++){
			b[i]=b[i]+conj(a(k,i))*a(k,i);
		}
		b[i]=sqrt(b[i]);
		for( int k=0 ; k<a.cols() ; k++){
			a(k,i)=a(k,i)/b[i];
		}
	}
}



