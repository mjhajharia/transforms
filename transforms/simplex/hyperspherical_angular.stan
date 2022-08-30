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
    real log_phi, phi, z, s, c;
    real s2_prod = 1;
    real log_halfpi = log(pi()) - log2();
    int rcounter = 2 * N - 3;
    for (i in 1:(N-1)) {
      z = log_inv_logit(y[i]);
      log_phi = z + log_halfpi;
      phi = exp(log_phi);
      s = sin(phi);
      c = cos(phi);
      x[i] = s2_prod * c^2;
      s2_prod *= s^2;
      log_det_jacobian += log_phi + log1m_exp(z) + rcounter * log(s) + log(c);
      rcounter -= 2;
    }
    x[N] = s2_prod;
    log_det_jacobian += (N - 1) * log2();
  }
}
model {
  target += log_det_jacobian;
  target += target_density_lp(x, alpha);
}
