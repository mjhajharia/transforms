data {
  int<lower=0> N;
  int<lower=0, upper=1> dirichlet_target;

  vector<lower=0>[dirichlet_target ? 0 : N-1] mu;
  cov_matrix[dirichlet_target ? 1 : N-1] sigma;
  vector<lower=0>[dirichlet_target ? N : 0] alpha;
}
parameters {
  vector[N] y;
}
transformed parameters {
  simplex[N] x;
  real<lower=0> r = 0;
  real log_det_jacobian = 0;
  {
    vector[N] z;
    real log_u;
    for (i in 1:N) {
      log_u = std_normal_lcdf(y[i]);
      z[i] = exponential_log_qf(log_u);
      r += z[i];
      log_det_jacobian += std_normal_lpdf(y[i]);
    }
    x = exp(log(z) - log(r));
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