data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector[N - 1] y;
}
transformed parameters {
  simplex[N] x;
  real log_det_jacobian = 0;
  {
    real log_s2, log_c2;
    real log_s2_sum = 0;
    int rcounter = N - 1;
    vector [N-1] z;
    real suminvk = 0;
    for (k in 1:(N-1)) {
      z[N-k] = y[N-k] + suminvk;
      suminvk += inv(k);
    }
    for (i in 1:(N-1)) {
      log_s2 = log_inv_logit(z[i]);
      log_c2 = log1m_inv_logit(z[i]);
      x[i] = exp(log_s2_sum + log_c2);
      log_s2_sum += log_s2;
      log_det_jacobian += rcounter * log_s2 + log_c2;
      rcounter -= 1;
    }
    x[N] = exp(log_s2_sum);
  }
}
model {
  target += log_det_jacobian;
  target += target_density_lp(x, alpha);
}
