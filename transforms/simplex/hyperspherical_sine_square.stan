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
    real yi;
    real zi, zprod, zprod_new;
    zprod = 1;
    for (i in 1:(N-1)) {
      yi = y[i];
      zi = inv_logit(yi);
      zprod_new = zprod * zi;
      x[i] = zprod - zprod_new;
      zprod = zprod_new;
      log_det_jacobian += (N - i) * log_inv_logit(yi) + log1m_inv_logit(yi);
    }
    x[N] = zprod;
  }
}
model {
  target += log_det_jacobian;
  target += target_density_lp(x, alpha);
}
