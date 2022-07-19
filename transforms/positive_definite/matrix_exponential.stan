#include transform_functions.stan
data {
  int<lower=1> N;
  real<lower=N-1> nu;
  cov_matrix[N] Sigma;
}
parameters {
  vector[length_tri(N)] y;
}
transformed parameters {
  real logJ = 0;
  matrix[N,N] Y = symmetrize_from_vec(y, N);
  vector[N] lambda = eigenvalues_sym(Y);
  matrix[N,N] U = eigenvectors_sym(Y);
  vector<lower=0>[N] exp_lambda;
  {
    real lambda_i;
    real lambda_j;
    real lambda_diff;
    for (j in 1:N) {
      lambda_j = lambda[j];
      exp_lambda[j] = exp(lambda_j);
      logJ += j * lambda_j;
      for (i in 1:(j - 1)) {
        lambda_i = lambda[i];
        if (lambda_i != lambda_j) {
          lambda_diff = lambda_j - lambda_i;
          logJ += log1m_exp(-lambda_diff) - log(lambda_diff);
        }
      }
    }
  }
  cov_matrix[N] X = quad_form_sym(diag_matrix(exp_lambda), U');
}
model {
  target += logJ;
  target += wishart_lupdf(X | nu, Sigma);
}
