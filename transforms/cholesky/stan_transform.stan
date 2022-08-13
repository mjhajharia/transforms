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
  real log_det_jacobian = 0;
  {
    int counter = 1;
    real sum_sqs;
    vector[choose_2(K)] z = tanh(y);
    
    for (i in 2 : K) {
      L[i, 1] = z[counter];
      counter += 1;
      sum_sqs = square(L[i, 1]);
      for (j in 2 : (i - 1)) {
        log_det_jacobian += 0.5 * log1m(sum_sqs);
        L[i, j] = z[counter] * sqrt(1 - sum_sqs);
        counter += 1;
        sum_sqs = sum_sqs + square(L[i, j]);
      }
      L[i, i] = sqrt(1 - sum_sqs);
    }
  }
}
model {
  target += log_det_jacobian;
  target += lkj_corr_cholesky_lpdf(L | 2);
}
