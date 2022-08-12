data {
  int<lower=2> N;
}
parameters {
  vector[N] y;
}
transformed parameters {
  real log_det_correction;
  unit_vector[N] x;
  real<lower=0> r;
  {
    real r2 = dot_self(y);
    r = sqrt(r2);
    // equivalent to (1-N)*log(r) + chi_square_lupdf(r2 | N)
    log_det_correction = -r2/2;
  }
  x = y / r;
}
model {
  target += log_det_correction;
}
