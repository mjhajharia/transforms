data {
  int<lower=2> N;
}
parameters {
  vector<lower=0,upper=pi()>[N-2] phi;
  vector[2] ylast;
}
transformed parameters {
  real log_det_correction = 0;
  // represent phi_{N-1} in (0, 2pi] with parameter expansion
  // cslast = [cos(phi_{N-1}), sin(phi_{N-1})]
  unit_vector[2] cslast
  real<lower=0> rlast;
  {
    real r2 = dot_self(ylast);
    rlast = sqrt(r2);
    // equivalent to (1-N)*log(rlast) + chi_square_lupdf(r2 | N)
    log_det_correction += -r2/2;
  }
  cslast = ylast / rlast;
  unit_vector[N] x;
  {
    real s_i;
    real c_i;
    real phi_i;
    real sin_prod = 1;
    int rcounter = N - 2;
    for (i in 1:(N-2)) {
      phi_i = phi[i];
      s_i = sin(phi_i);
      c_i = cos(phi_i);
      x[i] = sin_prod * c_i;
      sin_prod *= s_i;
      log_det_correction += rcounter * log(s_i);
      rcounter -= 1;
    }
    x[N-1] = sin_prod * cslast[1];
    x[N] = sin_prod * cslast[2];
  }
}
model {
  target += log_det_correction;
}
