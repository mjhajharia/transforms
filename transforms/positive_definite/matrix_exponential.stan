#include transform_functions.stan
data {
  int<lower=1> N;
}
parameters {
  vector[length_tri(N)] y;
}
transformed parameters {
  matrix[N,N] Y = symmetrize_from_vec(y, N);
  ordered[N] lambda = eigenvalues_sym(Y);
  matrix[N,N] U = eigenvectors_sym(Y);
  positive_ordered[N] exp_lambda;
  real log_det_jacobian = 0;
  {
    real lambda_i;
    real lambda_j;
    real lambda_diff;
    for (j in 1:N) {
      lambda_j = lambda[j];
      exp_lambda[j] = exp(lambda_j);
      log_det_jacobian += j * lambda_j;
      for (i in 1:(j - 1)) {
        lambda_i = lambda[i];
        if (lambda_i != lambda_j) {
          lambda_diff = lambda_j - lambda_i;
          log_det_jacobian += log1m_exp(-lambda_diff) - log(lambda_diff);
        }
      }
    }
  }
  cov_matrix[N] X = quad_form_sym(diag_matrix(exp_lambda), U');
}
model {
  target += log_det_jacobian;
}
