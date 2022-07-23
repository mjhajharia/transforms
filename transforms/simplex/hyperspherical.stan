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
    real phi_i, s_i, c_i;
    real sin2_prod = 1;
    int rcounter = 2 * N - 3;
    for (i in 1:(N-1)) {
      phi_i = phi[i];
      s_i = sin(phi_i);
      c_i = cos(phi_i);
      x[i] = sin2_prod * c_i^2;
      sin2_prod *= s_i^2;
      log_det_jacobian += rcounter * log(s_i) + log(c_i);
      rcounter -= 2;
    }
    x[N] = sin2_prod;
    // omit a (N - 1) * log(2) term, since it is constant
  }
}
model {
  target += log_det_jacobian;
  target += target_density_lp(x, alpha);
}
