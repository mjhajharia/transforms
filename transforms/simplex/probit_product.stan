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
    real zi, zprod = 1, zprod_new;
  
    for (i in 1:(N-1)) {
      yi = y[i];
      zi = Phi(yi);
      log_det_jacobian += std_normal_lpdf(yi |);
      zprod_new = zprod * zi;
      log_det_jacobian += log(zprod);
      x[i] = zprod - zprod_new;
      zprod = zprod_new;
    }
    x[N] = zprod;
  }
}
model {
  target += log_det_jacobian; 
  target += target_density_lp(x, alpha);
}
