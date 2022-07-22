data {
  int<lower=0> N;
  vector<lower=0>[N] alpha;
}
parameters {
  vector<lower=0,upper=pi()/2>[N-1] phi;
}
transformed parameters {
  simplex[N] x;
  real log_det_jacobian = 0;
  {
    real s2_i, c_i, phi_i;
    real sin2_prod = 1;
    int rcounter = 2 * N - 3;
    for (i in 1:(N-1)) {
      phi_i = phi[i];
      s2_i = sin(phi_i)^2;
      x[i] = sin2_prod * (1 - s2_i);
      sin2_prod *= s2_i;
      log_det_jacobian += rcounter * log(s2_i) + log1m(s2_i);
      rcounter -= 2;
    }
    x[N] = sin2_prod;
    log_det_jacobian /= 2;
  }
}
model {
  target += log_det_jacobian;
  target += target_density_lp(x, alpha);
}
