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
    real log_z, log_w;
    real s = 0;
    for (i in 1:(N-1)) {
      log_w = log_inv_logit(y[i]);
      log_z = log_w / (N - i);
      x[i] = exp(s + log1m_exp(log_z));
      s += log_z;
      log_det_jacobian += log_w + log1m_exp(log_w);
    }
    x[N] = exp(s);
  }
}
model {
  target += log_det_jacobian;
  target += target_density_lp(x, alpha);
}
