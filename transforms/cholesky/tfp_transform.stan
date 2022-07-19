#include transform_functions.stan
data {
  int<lower=0> K;
}
parameters {
  // y is a vector K-choose-2 unconstrained parameters
  vector[choose_2(K)] y;
}
transformed parameters {
  // L is a Cholesky factor of a K x K correlation matrix
  cholesky_factor_corr[K] L = diag_matrix(rep_vector(1, K));
  real log_det_jacobian;
  {
    int counter = 1;
    
    for (i in 2 : K) {
      L[i, 1] = y[counter];
      counter += 1;
      for (j in 2 : (i - 1)) {
        L[i, j] = y[counter];
        counter += 1;
      }
      L[i,  : i] = L[i,  : i] / sqrt(sum(square(L[i,  : i])));
      log_det_jacobian += (K - i + 1) * log(L[i, i]);
    }
  }
}
model {
  target += log_det_jacobian;
  target += lkj_corr_cholesky_lpdf(L | 2);
}
