data {
  int<lower=0> N;
  int<lower=0, upper=1> dirichlet_target;

  vector<lower=0>[dirichlet_target ? 0 : N-1] mu;
  cov_matrix[dirichlet_target ? 1 : N-1] sigma;
  vector<lower=0>[dirichlet_target ? N : 0] alpha;
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x;
  real log_det_jacobian = -lgamma(N);
  {
    real log_u, log_z;
    real sum_log_z = 0;
    for (i in 1:(N-1)) {
      log_u = log_inv_logit(y[i]);
      log_z = log_u / (N - i);
      x[i] = exp(sum_log_z + log1m_exp(log_z));
      sum_log_z += log_z;
      log_det_jacobian += log_u + log1m_exp(log_u);
    }
    x[N] = exp(sum_log_z);
  }
}
model {
  target += log_det_jacobian;
  if (dirichlet_target==1){
      target += target_density_lp(x, alpha);
  }
  if (dirichlet_target==0){
      x ~ multi_logit_normal(mu, sigma);
  }

  }