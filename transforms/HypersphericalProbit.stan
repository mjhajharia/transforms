data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
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
      log_u = std_normal_lcdf(y[i] |);
      log_z = log_u / (N - i);
      x[i] = exp(sum_log_z + log1m_exp(log_z));
      sum_log_z += log_z;
      log_det_jacobian += std_normal_lpdf(y[i] |);
    }
    x[N] = exp(sum_log_z);
  }
}
model {
  target += log_det_jacobian; 
  target += target_density_lp(x, alpha);
}