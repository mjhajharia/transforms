#include transform_functions.stan
data {
  int<lower=1> N;
}
parameters {
  vector[length_tri(N)] y;
}
transformed parameters {
  real log_det_jacobian = 0;
  cholesky_factor_cov[N] L;
  {
    int s = N + 1;
    int k = 1;
    real yii;
    for (i in 1:N) {
      for (j in 1:(i-1)) {
        L[i,j] = y[k];
        k += 1;
      }
      yii = y[k];
      L[i,i] = exp(yii);
      k += 1;
      log_det_jacobian += s * yii;
      s -= 1;
      for (j in (i+1):N)
        L[i,j] = 0;
    }
  }
  cov_matrix[N] X = multiply_lower_tri_self_transpose(L);
}
model {
  target += log_det_jacobian;
}
