#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace std;
using namespace arma;

RcppExport SEXP createL(SEXP Matrix_) {
  try {
    NumericMatrix Matrix(Matrix_);
    //int m = Matrix.ncol();
    int k = Matrix.nrow();
    mat MatrixA(Matrix.begin(), Matrix.nrow(), Matrix.ncol());
    mat K(k,k); K.zeros();
    for (int i=0; i < k; ++i) {
      for(int j=i; j < k; ++j) {
	mat diff = MatrixA.row(i)-MatrixA.row(j);
	K(i,j) = -sqrt(dot(diff,diff));
      }
    }
    K = K+K.t();
    return wrap(K);
  } catch (std::exception& e) {
    ::Rf_error( e.what());
  } catch (...) {
    ::Rf_error("unknown exception");
  }
}
  
